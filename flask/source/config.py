class Config:
    PROPAGATE_EXCEPTIONS = True

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True

config  = {
    "development"   : DevelopmentConfig,
    "production"    : ProductionConfig
}
