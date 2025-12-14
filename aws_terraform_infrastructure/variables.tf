variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-central-1" # Frankfurt (Common for BSH/German companies)
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "bsh-antigravity"
}
