variable "environment" {}
variable "project_name" {}
variable "vpc_id" {}
variable "subnet_ids" { type = list(string) }

# --- SageMaker Notebook Instance ---
resource "aws_sagemaker_notebook_instance" "ds_research" {
  name          = "${var.project_name}-notebook-${var.environment}"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t3.medium"
  subnet_id     = var.subnet_ids[0]

  tags = {
    Name = "Data Science Research Sandbox"
  }
}

# --- IAM Role for SageMaker ---
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
