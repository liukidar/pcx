container=$(which podman 2>/dev/null || which docker)

$container run --rm -p 6379:6379 -v redis-data:/data  docker.io/redis:latest
