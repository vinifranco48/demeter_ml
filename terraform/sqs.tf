resource "aws_sqs_queue" "processing_dlq" {
  name = "${var.project_name}-processing-dlq-${var.environment}"
  tags = var.tags
}

resource "aws_sqs_queue" "processing_queue" {
  name                      = "${var.project_name}-processing-queue-${var.environment}"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 86400 # 1 day
  receive_wait_time_seconds = 10
  
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.processing_dlq.arn
    maxReceiveCount     = 3
  })

  tags = var.tags
}
