# AWS Deployment Guide for MAB Web Testing

This guide provides step-by-step instructions for deploying the MAB Web Testing application to AWS.

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Python 3.12+ installed
- Git installed

## Deployment Options

The application can be deployed to AWS in several ways:

1. **AWS Elastic Beanstalk** (recommended for simplicity)
2. **Amazon EC2** with manual setup
3. **AWS Lambda** with API Gateway (serverless approach)

This guide focuses on the Elastic Beanstalk deployment, which is the simplest approach.

## Step 1: Prepare Your Application

1. Make sure your application is working locally
2. Create or update your `requirements.txt` file
3. Create a `.ebignore` file to exclude unnecessary files

```
# .ebignore
venv/
.mab_simulator/
__pycache__/
*.pyc
.env
.git
.gitignore
```

4. Create a `Procfile` in the root directory:

```
# Procfile
web: python app.py
```

## Step 2: Initialize Elastic Beanstalk

1. Install the EB CLI:

```bash
pip install awsebcli
```

2. Initialize EB application:

```bash
eb init -p python-3.12 mab-web-testing
```

3. When prompted, select your region and create or select credentials

## Step 3: Configure the Environment

1. Create an Elastic Beanstalk environment:

```bash
eb create mab-testing-env
```

2. Configure environment variables via the AWS Console:
   - Go to Elastic Beanstalk > Your Environment > Configuration
   - Add environment variables:
     - `AWS_ACCESS_KEY_ID`: Your AWS access key
     - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
     - `AWS_REGION`: Your AWS region
     - `MAB_ALGORITHM`: The MAB algorithm to use
     - Any other configuration variables your application needs

## Step 4: Deploy the Application

1. Deploy your application:

```bash
eb deploy
```

2. Open the application:

```bash
eb open
```

## Step 5: Set Up AWS Resources

Your application needs several AWS resources. Use the script below to set them up:

```bash
python setup_aws_resources.py
```

This script will:
- Create an S3 bucket for storing results
- Create a DynamoDB table for tracking interactions
- Configure the necessary IAM permissions

## Step 6: Configure Monitoring

1. Set up CloudWatch alarms:
   - Go to CloudWatch in AWS Console
   - Create alarms for key metrics:
     - CPU usage
     - Memory usage
     - Application errors

2. Set up a CloudWatch dashboard for visualization:
   - Create a new dashboard
   - Add widgets for key metrics

## Step 7: Test Your Deployment

1. Visit your application URL
2. Interact with the different page designs
3. Check the logs:

```bash
eb logs
```

4. Monitor AWS resources in the console

## Troubleshooting

- **Application doesn't start**: Check the logs with `eb logs`
- **Permission errors**: Verify IAM roles and policies
- **Database errors**: Check DynamoDB table setup
- **High costs**: Review resource usage and scaling settings

## Updating Your Application

To update your application after making changes:

```bash
eb deploy
```

## Cleaning Up

To avoid unnecessary AWS charges, terminate resources when done:

```bash
eb terminate mab-testing-env
```

Then manually delete:
- S3 buckets
- DynamoDB tables
- Any other resources created for the application 