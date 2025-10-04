import os
from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    # config settings
    app.config.from_mapping(
        SECRET_KEY = "dev" # UPDATE LATER
    )

    # ensure the instance folder exists
    # try:
    #     os.makedirs(app.instance_path)
    # except OSError:
    #     pass

    if test_config is None:
        app.config.from_pyfile("config.py")
    else:
        app.config.from_mapping(test_config)
    

    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app
