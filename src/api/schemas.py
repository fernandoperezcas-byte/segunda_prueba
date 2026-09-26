from pydantic import BaseModel, Field

class HousingInput(BaseModel):
    MedInc: float = Field(..., description="Ingreso medio en el bloque (en decenas de miles de dólares)", example=3.5)
    HouseAge: float = Field(..., description="Edad media de la casa en el bloque", example=15.0)
    AveRooms: float = Field(..., description="Número promedio de habitaciones por hogar", example=5.2)
    AveBedrms: float = Field(..., description="Número promedio de dormitorios por hogar", example=1.0)
    Population: float = Field(..., description="Población del bloque", example=800.0)
    AveOccup: float = Field(..., description="Ocupación promedio por hogar", example=3.0)
    Latitude: float = Field(..., description="Latitud del bloque", example=34.05)
    Longitude: float = Field(..., description="Longitud del bloque", example=-118.24)
