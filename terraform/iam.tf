# Common Assume Role Policy for Lambdas
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

# --- Upload Lambda Role ---
resource "aws_iam_role" "upload_lambda_role" {
  name               = "${var.project_name}-upload-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
  tags               = var.tags
}

# Policy for Upload Lambda: Write to S3 Raw, Send to SQS, Logs
resource "aws_iam_policy" "upload_lambda_policy" {
  name        = "${var.project_name}-upload-policy-${var.environment}"
  description = "Allow upload lambda to write to S3 and SQS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
        ]
        Resource = "${aws_s3_bucket.raw_images.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
        ]
        Resource = aws_sqs_queue.processing_queue.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "upload_lambda_attach" {
  role       = aws_iam_role.upload_lambda_role.name
  policy_arn = aws_iam_policy.upload_lambda_policy.arn
}

# --- Processor Lambda Role ---
resource "aws_iam_role" "processor_lambda_role" {
  name               = "${var.project_name}-processor-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
  tags               = var.tags
}

# Policy for Processor Lambda: Read S3 Raw, Write S3 Processed, Receive SQS, Logs
resource "aws_iam_policy" "processor_lambda_policy" {
  name        = "${var.project_name}-processor-policy-${var.environment}"
  description = "Allow processor lambda to read/write S3 and consume SQS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
        ]
        Resource = "${aws_s3_bucket.raw_images.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
        ]
        Resource = "${aws_s3_bucket.processed_data.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.processing_queue.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "processor_lambda_attach" {
  role       = aws_iam_role.processor_lambda_role.name
  policy_arn = aws_iam_policy.processor_lambda_policy.arn
}
