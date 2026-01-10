# Publishing Huxley to Docker Hub

## Prerequisites

1. **Docker Hub Account**: Sign up at https://hub.docker.com
2. **Docker CLI**: Ensure Docker is installed and running
3. **Login to Docker Hub**:
   ```bash
   docker login
   # Enter your Docker Hub username and password
   ```

## Publishing Steps

### 1. Build the Docker Images

Build all variants of your image:

```bash
# Build production image
docker build --target production -t huxley-research/huxley:latest .
docker build --target production -t huxley-research/huxley:0.6.0 .

# Build onboarding image (with WebUI)
docker build --target onboarding -t huxley-research/huxley:onboarding .
docker build --target onboarding -t huxley-research/huxley:0.6.0-onboarding .

# Build development image (optional)
docker build --target development -t huxley-research/huxley:dev .
```

### 2. Tag Images Appropriately

Use semantic versioning and descriptive tags:

```bash
# Tag with version
docker tag huxley-research/huxley:latest huxley-research/huxley:0.6.0

# Tag onboarding variant
docker tag huxley-research/huxley:onboarding huxley-research/huxley:0.6.0-onboarding

# Optional: Tag with commit SHA for traceability
docker tag huxley-research/huxley:latest huxley-research/huxley:$(git rev-parse --short HEAD)
```

### 3. Push Images to Docker Hub

```bash
# Push latest
docker push huxley-research/huxley:latest
docker push huxley-research/huxley:0.6.0

# Push onboarding variant
docker push huxley-research/huxley:onboarding
docker push huxley-research/huxley:0.6.0-onboarding

# Push all tags at once
docker push huxley-research/huxley --all-tags
```

### 4. Test the Published Image

Verify your published image works:

```bash
# Pull and run the published image
docker pull huxley-research/huxley:latest
docker run -it --rm huxley-research/huxley:latest --help

# Test onboarding image
docker pull huxley-research/huxley:onboarding
docker run -p 3000:3000 huxley-research/huxley:onboarding
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Images

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: docker.io
  IMAGE_NAME: huxley-research/huxley

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push production image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: production
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=${{ env.IMAGE_NAME }}:buildcache
          cache-to: type=registry,ref=${{ env.IMAGE_NAME }}:buildcache,mode=max

      - name: Build and push onboarding image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: onboarding
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ${{ env.IMAGE_NAME }}:onboarding
            ${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}-onboarding
          labels: ${{ steps.meta.outputs.labels }}
```

### Setup GitHub Secrets

1. Go to your GitHub repository settings
2. Navigate to **Secrets and variables** → **Actions**
3. Add secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password or access token

## Docker Hub Repository Setup

### 1. Create Repository Description

On Docker Hub, add a comprehensive description:

```markdown
# Huxley - Biological AI Research Framework

Huxley is an advanced AI framework for biological research with multi-provider LLM support, specialized tools for molecular biology, and intelligent model orchestration.

## Quick Start

### With Interactive Onboarding (Recommended)
docker run -p 3000:3000 huxley-research/huxley:onboarding

Visit http://localhost:3000 to configure your setup.

### Direct Usage
docker run -it huxley-research/huxley:latest huxley chat

## Documentation
- GitHub: https://github.com/Huxley-Research/Huxley
- Docs: https://github.com/Huxley-Research/Huxley/blob/main/README.md
```

### 2. Add Tags

Organize with appropriate tags:
- `ai`
- `biology`
- `research`
- `llm`
- `scientific-computing`
- `molecular-biology`

### 3. Link GitHub Repository

Connect your Docker Hub repository to your GitHub repo for:
- Automated builds on push
- README sync
- Source code reference

## Multi-Architecture Builds

Support ARM64 (Apple Silicon, Raspberry Pi) and AMD64:

```bash
# Create and use a builder
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  -t huxley-research/huxley:latest \
  -t huxley-research/huxley:0.6.0 \
  --push \
  .

# Build onboarding variant
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target onboarding \
  -t huxley-research/huxley:onboarding \
  -t huxley-research/huxley:0.6.0-onboarding \
  --push \
  .
```

## Update docker-compose.yml for Published Images

Update your `docker-compose.yml` to use published images:

```yaml
services:
  huxley:
    image: huxley-research/huxley:onboarding  # Use published image
    # Remove 'build:' section when using published images
    ports:
      - "3000:3000"
    environment:
      - ONBOARDING_MODE=true
```

## Version Management

Follow semantic versioning:
- **Major** (1.0.0): Breaking changes
- **Minor** (0.6.0): New features, backward compatible
- **Patch** (0.6.1): Bug fixes

Tag releases consistently:
```bash
git tag -a v0.6.0 -m "Release version 0.6.0 with Docker onboarding"
git push origin v0.6.0
```

## Security Scanning

Scan images for vulnerabilities before publishing:

```bash
# Using Docker Scout (built-in)
docker scout cves huxley-research/huxley:latest

# Using Trivy
docker run aquasec/trivy image huxley-research/huxley:latest

# Using Snyk
snyk container test huxley-research/huxley:latest
```

## Publishing Checklist

- [ ] Build all image variants (production, onboarding, dev)
- [ ] Tag with semantic version numbers
- [ ] Test images locally
- [ ] Scan for security vulnerabilities
- [ ] Push to Docker Hub
- [ ] Update Docker Hub description and README
- [ ] Create GitHub release with changelog
- [ ] Update documentation with new version
- [ ] Announce release to users

## Alternative Registries

### GitHub Container Registry (ghcr.io)

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag and push
docker tag huxley-research/huxley:latest ghcr.io/huxley-research/huxley:latest
docker push ghcr.io/huxley-research/huxley:latest
```

### AWS ECR

```bash
# Login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag huxley-research/huxley:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/huxley:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/huxley:latest
```

## Maintenance

Regularly update base images and dependencies:

```bash
# Update base image in Dockerfile
FROM python:3.11-slim  # Check for newer versions

# Rebuild and republish
docker build --no-cache -t huxley-research/huxley:latest .
docker push huxley-research/huxley:latest
```

## Usage by End Users

After publishing, users can run Huxley with:

```bash
# Quick start with onboarding
docker run -p 3000:3000 huxley-research/huxley:onboarding

# Or with docker-compose
curl -O https://raw.githubusercontent.com/Huxley-Research/Huxley/main/docker-compose.yml
docker-compose up
```
