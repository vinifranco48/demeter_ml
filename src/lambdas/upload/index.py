import json
import boto3
import os
import uuid

s3 = boto3.client('s3')
sqs = boto3.client('sqs')

RAW_BUCKET = os.environ['RAW_BUCKET']
QUEUE_URL = os.environ['QUEUE_URL']

def handler(event, context):
    # TODO: Parse multipart/form-data or base64 body to get image
    # For now, assuming body is the image content or a JSON with image data
    
    job_id = str(uuid.uuid4())
    key = f"{job_id}.jpg"
    
    # Simulating saving to S3
    # s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=event['body'])
    
    # Send message to SQS
    message = {
        "job_id": job_id,
        "s3_key": key,
        "bucket": RAW_BUCKET
    }
    
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message)
    )
    
    return {
        "statusCode": 202,
        "body": json.dumps({
            "message": "Image accepted for processing",
            "job_id": job_id
        })
    }
