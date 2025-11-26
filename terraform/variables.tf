variable "project_name" {
  description = "Project name to be used as prefix for resources"
  type        = string
  default     = "demeter-ml"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {
    Project     = "DemeterML"
    ManagedBy   = "Terraform"
  }
}
