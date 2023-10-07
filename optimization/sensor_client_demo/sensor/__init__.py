def SensorClient(sensor_name="Azure/Flexx"):
    if sensor_name == "Azure":
        from .azure_client import AzureRGBDFetchNode
        return AzureRGBDFetchNode
    elif sensor_name == "Flexx":
        from .flexx_client import FlexxRGBDFetchNode
        return FlexxRGBDFetchNode
    else:
        raise ValueError("sensor_name is not right : ".format(sensor_name))