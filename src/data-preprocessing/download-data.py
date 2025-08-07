import cdsapi

dataset = "cams-gridded-solar-radiation"
request = {
    "variable": [
        "global_horizontal_irradiation",
        "direct_horizontal_irradiation",
        "diffuse_horizontal_irradiation",
        "direct_normal_irradiation"
    ],
    "sky_type": ["observed_cloud"],
    "version": ["4.6"],
    "year": [
        "2019"
    ],
    "month": [
        "01"
    ]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
