# Zip the upload lambda code
data "archive_file" "upload_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../src/lambdas/upload/index.py"
  output_path = "${path.module}/upload_lambda.zip"
}

# Zip the processor lambda code
data "archive_file" "processor_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../src/lambdas/processor/index.py"
  output_path = "${path.module}/processor_lambda.zip"
}

# Upload Lambda Function
resource "aws_lambda_function" "upload" {
  filename         = data.archive_file.upload_lambda_zip.output_path
  function_name    = "${var.project_name}-upload-${var.environment}"
  role             = aws_iam_role.upload_lambda_role.arn
  handler          = "index.handler"
  source_code_hash = data.archive_file.upload_lambda_zip.output_base64sha256
  runtime          = "python3.9"
  timeout          = 10

  environment {
    variables = {
      RAW_BUCKET = aws_s3_bucket.raw_images.id
      QUEUE_URL  = aws_sqs_queue.processing_queue.id
    }
  }

  tags = var.tags
}

# Processor Lambda Function
resource "aws_lambda_function" "processor" {
  filename         = data.archive_file.processor_lambda_zip.output_path
  function_name    = "${var.project_name}-processor-${var.environment}"
  role             = aws_iam_role.processor_lambda_role.arn
  handler          = "index.handler"
  source_code_hash = data.archive_file.processor_lambda_zip.output_base64sha256
  runtime          = "python3.9"
  timeout          = 60

  environment {
    variables = {
      PROCESSED_BUCKET = aws_s3_bucket.processed_data.id
    }
  }

  tags = var.tags
}

# SQS Trigger for Processor Lambda
resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = aws_sqs_queue.processing_queue.arn
  function_name    = aws_lambda_function.processor.arn
  batch_size       = 1
  enabled          = true
}
