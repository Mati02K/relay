package main

import (
	"context"
	"log"
	"net"
	"os"
	"strings"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"

	pb "relay/membership-etcd/membership"
)

// server implements the MembershipService gRPC interface.
type server struct {
	pb.UnimplementedMembershipServiceServer
	etcdClient *clientv3.Client
	nodeID     string
}

// Register writes a node entry under /relay/nodes/{nodeId} in etcd.
func (s *server) Register(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	_, err := s.etcdClient.Put(ctx, "/relay/nodes/"+req.NodeId, req.MetadataJson)
	if err != nil {
		return &pb.RegisterResponse{Success: false}, err
	}
	return &pb.RegisterResponse{Success: true}, nil
}

// Deregister deletes a node's entry from etcd.
func (s *server) Deregister(ctx context.Context, req *pb.NodeIdRequest) (*pb.Empty, error) {
	_, err := s.etcdClient.Delete(ctx, "/relay/nodes/"+req.NodeId)
	return &pb.Empty{}, err
}

// GetAliveMembers returns all nodes currently registered under /relay/nodes/.
func (s *server) GetAliveMembers(ctx context.Context, _ *pb.Empty) (*pb.MemberList, error) {
	resp, err := s.etcdClient.Get(ctx, "/relay/nodes/", clientv3.WithPrefix())
	if err != nil {
		return nil, err
	}
	nodes := make([]*pb.Node, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		id := strings.TrimPrefix(string(kv.Key), "/relay/nodes/")
		nodes = append(nodes, &pb.Node{
			Id:           id,
			MetadataJson: string(kv.Value),
		})
	}
	return &pb.MemberList{Nodes: nodes}, nil
}

// WatchMembership streams JOIN/LEFT events as nodes are added or removed from etcd.
func (s *server) WatchMembership(_ *pb.Empty, stream pb.MembershipService_WatchMembershipServer) error {
	watchCh := s.etcdClient.Watch(stream.Context(), "/relay/nodes/", clientv3.WithPrefix())
	for {
		select {
		case <-stream.Context().Done():
			return nil
		case watchResp, ok := <-watchCh:
			if !ok {
				return nil
			}
			for _, event := range watchResp.Events {
				id := strings.TrimPrefix(string(event.Kv.Key), "/relay/nodes/")
				eventType := pb.MemberEvent_JOINED
				if event.Type == clientv3.EventTypeDelete {
					eventType = pb.MemberEvent_LEFT
				}
				if err := stream.Send(&pb.MemberEvent{
					Type: eventType,
					Node: &pb.Node{
						Id:           id,
						MetadataJson: string(event.Kv.Value),
					},
				}); err != nil {
					return err
				}
			}
		}
	}
}

// HoldLeadership campaigns for leadership using a per-stream session. The caller
// receives a LeaderStatus message when it wins. When the gRPC stream closes
// (coordinator crash or graceful shutdown) the session expires and the next
// candidate wins the election immediately.
func (s *server) HoldLeadership(_ *pb.Empty, stream pb.MembershipService_HoldLeadershipServer) error {
	ctx := stream.Context()

	// Short TTL so an ungraceful crash is detected quickly by etcd.
	session, err := concurrency.NewSession(s.etcdClient, concurrency.WithTTL(10))
	if err != nil {
		return err
	}

	election := concurrency.NewElection(session, "/relay/election")

	// Campaign blocks until this node wins the election.
	if err := election.Campaign(ctx, s.nodeID); err != nil {
		session.Close()
		return err
	}

	// Notify the caller that it is now the leader.
	if err := stream.Send(&pb.LeaderStatus{IsLeader: true, LeaderId: s.nodeID}); err != nil {
		election.Resign(context.Background())
		session.Close()
		return err
	}

	// Hold the stream open — keeping the session alive — until the caller disconnects.
	<-ctx.Done()

	// Resign immediately so the next candidate wins without waiting for TTL expiry.
	election.Resign(context.Background())
	session.Close()
	return nil
}

// HoldLease creates an etcd session with the requested TTL and keeps the
// underlying lease alive for as long as the stream is open. The first message
// carries the lease id so the client can attach PutWithLease writes to it.
// When the stream closes — graceful client shutdown or transport failure —
// the session is revoked and etcd auto-deletes every key bound to the lease.
//
// This is the worker-liveness primitive: a crashed worker stops the stream,
// and within ~TTL seconds its registration and telemetry keys disappear.
func (s *server) HoldLease(req *pb.LeaseRequest, stream pb.MembershipService_HoldLeaseServer) error {
	ctx := stream.Context()

	ttl := req.TtlSeconds
	if ttl < 5 {
		ttl = 5
	}
	if ttl > 60 {
		ttl = 60
	}

	session, err := concurrency.NewSession(s.etcdClient, concurrency.WithTTL(int(ttl)))
	if err != nil {
		return err
	}

	leaseID := int64(session.Lease())
	if err := stream.Send(&pb.LeaseStatus{LeaseId: leaseID, Alive: true}); err != nil {
		session.Close()
		return err
	}

	// Hold the stream open until the client disconnects.
	<-ctx.Done()
	session.Close()
	return nil
}

// PutWithLease writes a key bound to the given lease id. The key expires
// automatically when the lease expires or is revoked.
func (s *server) PutWithLease(ctx context.Context, req *pb.LeaseKVPair) (*pb.Empty, error) {
	_, err := s.etcdClient.Put(ctx, req.Key, req.Value, clientv3.WithLease(clientv3.LeaseID(req.LeaseId)))
	return &pb.Empty{}, err
}

// GetByPrefix returns all key-value pairs whose keys start with the given prefix.
func (s *server) GetByPrefix(ctx context.Context, req *pb.KeyRequest) (*pb.KVList, error) {
	resp, err := s.etcdClient.Get(ctx, req.Key, clientv3.WithPrefix())
	if err != nil {
		return nil, err
	}
	items := make([]*pb.KVPair, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		items = append(items, &pb.KVPair{
			Key:   string(kv.Key),
			Value: string(kv.Value),
		})
	}
	return &pb.KVList{Items: items}, nil
}

// Put writes an arbitrary key-value pair to etcd.
func (s *server) Put(ctx context.Context, req *pb.KVPair) (*pb.Empty, error) {
	_, err := s.etcdClient.Put(ctx, req.Key, req.Value)
	return &pb.Empty{}, err
}

// Get retrieves a value from etcd by key; sets Found=false if the key does not exist.
func (s *server) Get(ctx context.Context, req *pb.KeyRequest) (*pb.ValueResponse, error) {
	resp, err := s.etcdClient.Get(ctx, req.Key)
	if err != nil {
		return nil, err
	}
	if len(resp.Kvs) == 0 {
		return &pb.ValueResponse{Found: false}, nil
	}
	return &pb.ValueResponse{
		Value: string(resp.Kvs[0].Value),
		Found: true,
	}, nil
}

func main() {
	endpoints := strings.Split(getEnv("ETCD_ENDPOINTS", "localhost:2379"), ",")
	grpcPort := getEnv("GRPC_PORT", "50051")
	nodeID := getEnv("NODE_ID", "membership-etcd")

	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatalf("etcd connect failed: %v", err)
	}
	defer cli.Close()

	lis, err := net.Listen("tcp", ":"+grpcPort)
	if err != nil {
		log.Fatalf("listen failed: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterMembershipServiceServer(grpcServer, &server{
		etcdClient: cli,
		nodeID:     nodeID,
	})

	log.Printf("membership-etcd listening on :%s (etcd: %v)", grpcPort, endpoints)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve failed: %v", err)
	}
}

// getEnv returns the value of an environment variable or a default if unset.
func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
