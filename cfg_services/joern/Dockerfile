FROM alpine:3.19

# Install dependencies
RUN apk add --no-cache \
    bash \
    curl \
    wget \
    git \
    openjdk17 \
    python3 \
    py3-pip \
    unzip

# Install Joern
WORKDIR /opt
RUN wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh && \
    chmod +x joern-install.sh && \
    ./joern-install.sh --interactive=false && \
    rm joern-install.sh

# Add Joern to PATH
ENV PATH="/opt/joern/joern-cli:$PATH"

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages flask requests pydot

# Create app directory
WORKDIR /app

# Copy Flask application
COPY app.py /app/

# Expose Flask port
EXPOSE 5000

# Run Flask directly
CMD ["python3", "/app/app.py"]