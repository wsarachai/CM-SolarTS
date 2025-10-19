import cdsapi

dataset = "derived-utci-historical"
request = {
    "variable": ["mean_radiant_temperature"],
    "version": "1_1",
    "product_type": "consolidated_dataset",
    "year": [
        "2020", "2021", "2022",
        "2023", "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "area": [18.903, 99.01, 18.893, 99.015]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
