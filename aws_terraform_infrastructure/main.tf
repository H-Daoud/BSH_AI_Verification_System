module "networking" {
  source = "./modules/networking"

  environment  = var.environment
  project_name = var.project_name
  vpc_cidr     = "10.0.0.0/16"
}

module "storage" {
  source = "./modules/storage"

  environment  = var.environment
  project_name = var.project_name
}

module "ml_ops" {
  source = "./modules/ml_ops"

  environment  = var.environment
  project_name = var.project_name
  vpc_id       = module.networking.vpc_id
  subnet_ids   = module.networking.private_subnet_ids
}

module "compute" {
  source = "./modules/compute"

  environment      = var.environment
  project_name     = var.project_name
  vpc_id           = module.networking.vpc_id
  public_subnets   = module.networking.public_subnet_ids
  private_subnets  = module.networking.private_subnet_ids
  ecr_repo_url     = module.storage.ecr_repo_url
}
