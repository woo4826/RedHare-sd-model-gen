from flask import Flask

from config import config

app = Flask(import_name=__name__)

def create_app():
    app.config.from_object(obj=config["development"])

    with app.app_context():
        from cmg_gen.index import bp_index

        app.register_blueprint(blueprint=bp_index)
        
        #path setting
        UPLOAD_FOLDER = '/workspace/uploads'
        MODEL_OUTPUT_FOLDER = '/workspace/output'
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        app.config['MODEL_OUTPUT_FOLDER'] = MODEL_OUTPUT_FOLDER
        app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS


        return app