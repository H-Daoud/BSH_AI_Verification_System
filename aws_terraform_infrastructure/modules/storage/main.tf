variable "environment" {}
variable "project_name" {}

# --- S3 Buckets ---
resource "aws_s3_bucket" "data_raw" {
  bucket = "${var.project_name}-data-raw-${var.environment}"
}

resource "aws_s3_bucket" "data_processed" {
  bucket = "${var.project_name}-data-processed-${var.environment}"
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${var.environment}"
}

# --- ECR Repositories ---
resource "aws_ecr_repository" "api_service" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

output "ecr_repo_url" {
  value = aws_ecr_repository.api_service.repository_url
}

output "model_bucket_arn" {
  value = aws_s3_bucket.models.arn
}
