.PHONY: proto go-setup lint typecheck

proto:
	@echo "Generating Python gRPC stubs..."
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/membership \
		--grpc_python_out=src/membership \
		proto/relay.proto
	@# grpc_tools generates absolute imports; fix to relative for package use.
	sed -i 's/^import relay_pb2/from . import relay_pb2/' src/membership/relay_pb2_grpc.py

	@echo "Generating Go gRPC stubs..."
	mkdir -p src/membership/etcd-go/membership
	protoc \
		-I proto \
		--go_out=src/membership/etcd-go/membership \
		--go_opt=paths=source_relative \
		--go-grpc_out=src/membership/etcd-go/membership \
		--go-grpc_opt=paths=source_relative \
		proto/relay.proto

go-setup:
	cd src/membership/etcd-go && \
		go get google.golang.org/grpc@latest \
		go.etcd.io/etcd/client/v3@latest \
		google.golang.org/protobuf@latest && \
		go mod tidy

lint:
	ruff check . && ruff format --check .

typecheck:
	mypy src/
