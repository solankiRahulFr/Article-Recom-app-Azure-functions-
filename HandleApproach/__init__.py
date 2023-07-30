import azure.functions as func
from FlaskApp import app

def main(req: func.HttpRequest, inputBlob: func.InputStream, context: func.Context) -> func.HttpResponse:
    """Each request is redirected to the WSGI handler.
    """
    return func.WsgiMiddleware(app.wsgi_app).handle(req, inputBlob, context)
