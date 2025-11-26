resource "aws_s3_bucket" "raw_images" {
  bucket = "${var.project_name}-raw-images-${var.environment}"
  
  tags = var.tags
}

resource "aws_s3_bucket" "processed_data" {
  bucket = "${var.project_name}-processed-data-${var.environment}"

  tags = var.tags
}

# Block public access for security
resource "aws_s3_bucket_public_access_block" "raw_images" {
  bucket = aws_s3_bucket.raw_images.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "processed_data" {
  bucket = aws_s3_bucket.processed_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
