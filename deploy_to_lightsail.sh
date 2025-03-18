#!/bin/bash
set -e

# Variables (customize these)
LIGHTSAIL_INSTANCE_NAME="mab-web-app"
REGION="us-east-1"
BLUEPRINT="amazon-linux-2"
BUNDLE_ID="nano_3_0"  # smallest instance

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Create Lightsail instance
echo "Creating Lightsail instance..."
aws lightsail create-instances \
    --instance-names "$LIGHTSAIL_INSTANCE_NAME" \
    --availability-zone "${REGION}a" \
    --blueprint-id "$BLUEPRINT" \
    --bundle-id "$BUNDLE_ID" \
    --user-data "#!/bin/bash
# Install Docker and Docker Compose
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user
sudo curl -L \"https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create app directory
mkdir -p /home/ec2-user/mab-app
cd /home/ec2-user/mab-app

# Clone the repository (replace with your repository URL)
git clone https://github.com/your-username/your-repo.git .

# Set up environment variables
cp .env.example .env

# Start the application with Docker Compose
docker-compose up -d
"

# Wait for the instance to be running
echo "Waiting for instance to become available..."
aws lightsail wait instance-running --instance-name "$LIGHTSAIL_INSTANCE_NAME"

# Get public IP
PUBLIC_IP=$(aws lightsail get-instance --instance-name "$LIGHTSAIL_INSTANCE_NAME" --query 'instance.publicIpAddress' --output text)

# Open ports
echo "Opening ports for HTTP and HTTPS..."
aws lightsail open-instance-public-ports \
    --port-info fromPort=80,toPort=80,protocol=TCP \
    --instance-name "$LIGHTSAIL_INSTANCE_NAME"

aws lightsail open-instance-public-ports \
    --port-info fromPort=443,toPort=443,protocol=TCP \
    --instance-name "$LIGHTSAIL_INSTANCE_NAME"

aws lightsail open-instance-public-ports \
    --port-info fromPort=5000,toPort=5000,protocol=TCP \
    --instance-name "$LIGHTSAIL_INSTANCE_NAME"

echo "Lightsail instance created with IP: $PUBLIC_IP"
echo "Your MAB Web App will be available at http://$PUBLIC_IP:5000 once setup completes."
echo "It may take a few minutes for the user data script to finish executing." 