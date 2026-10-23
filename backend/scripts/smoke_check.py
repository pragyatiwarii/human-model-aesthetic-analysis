from app.core.config import settings
from app.services.agiqa3k import status

print("Database:", settings.db_path)
print("Confidence threshold:", settings.detection_confidence)
print("Dataset status:")
print(status())
