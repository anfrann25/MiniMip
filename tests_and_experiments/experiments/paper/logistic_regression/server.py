from mini_mip_system.server.grpc_agg_server import serve
import asyncio

available_clients=3

if __name__ == "__main__":
    asyncio.run(serve(available_clients=available_clients))

