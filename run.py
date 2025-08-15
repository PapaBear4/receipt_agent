from app.main import app
from app.config import settings
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)
