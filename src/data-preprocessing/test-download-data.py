import cdsapi

client = cdsapi.Client()

dataset = 'cams-global-radiative-forcings'
request = {
  'variable': ['radiative_forcing_of_carbon_dioxide'],
  'forcing_type': 'instantaneous',
  'band': ['long_wave'],
  'sky_type': ['all_sky'],
  'level': ['surface'],
  'version': ['2'],
  'year': ['2018'],
  'month': ['06']
}
target = 'download.grib'

client.retrieve(dataset, request, target)