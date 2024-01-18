import folium
from folium import plugins
import geopandas as gpd

# Dane z wyników
results_data = {
    'context': {'geo_bounds': {'circle': {'center': {'latitude': 51.117045, 'longitude': 17.001045}, 'radius': 22000}}},
    'results': [
        {
            'categories': [
                {'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/hotdog_', 'suffix': '.png'},
                 'id': 13058,
                 'name': 'Hot Dog Joint',
                 'plural_name': 'Hot Dog Joints',
                 'short_name': 'Hot Dogs'},
                {'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_', 'suffix': '.png'},
                 'id': 13034,
                 'name': 'Café',
                 'plural_name': 'Cafés',
                 'short_name': 'Café'}
            ],
            'chains': [],
            'closed_bucket': 'VeryLikelyOpen',
            'distance': 119,
            'fsq_id': '5ec3548ab342b000081e5438',
            'geocodes': {'main': {'latitude': 51.11788, 'longitude': 17.000108},
                         'roof': {'latitude': 51.11788, 'longitude': 17.000108}},
            'link': '/v3/places/5ec3548ab342b000081e5438',
            'location': {'address': 'Słubicka 18',
                         'country': 'PL',
                         'cross_street': '',
                         'formatted_address': 'Słubicka 18, 53-615 Wrocław',
                         'locality': 'Wrocław',
                         'postcode': '53-615',
                         'region': 'Województwo dolnośląskie'},
            'name': 'Wild Bean Cafe',
            'related_places': {},
            'timezone': 'Europe/Warsaw'
        },
        {
            'categories': [
                {'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_', 'suffix': '.png'},
                 'id': 13035,
                 'name': 'Coffee Shop',
                 'plural_name': 'Coffee Shops',
                 'short_name': 'Coffee Shop'},
                {'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/italian_', 'suffix': '.png'},
                 'id': 13236,
                 'name': 'Italian Restaurant',
                 'plural_name': 'Italian Restaurants',
                 'short_name': 'Italian'},
                {'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/deli_', 'suffix': '.png'},
                 'id': 13334,
                 'name': 'Sandwich Spot',
                 'plural_name': 'Sandwich Spots',
                 'short_name': 'Sandwich Spot'}
            ],
            'chains': [],
            'closed_bucket': 'VeryLikelyOpen',
            'distance': 452,
            'fsq_id': '5e355c93e6d2380008e29b16',
            'geocodes': {'main': {'latitude': 51.112956, 'longitude': 17.00083},
                         'roof': {'latitude': 51.112956, 'longitude': 17.00083}},
            'link': '/v3/places/5e355c93e6d2380008e29b16',
            'location': {'address': 'Strzegomska 3b-3c',
                         'country': 'PL',
                         'cross_street': '',
                         'formatted_address': 'Strzegomska 3b-3c, 53-611 Wrocław',
                         'locality': 'Wrocław',
                         'postcode': '53-611',
                         'region': 'Dolnośląskie'},
            'name': 'Coffee & Dreams',
            'related_places': {},
            'timezone': 'Europe/Warsaw'
        }
    ]
}
# Wrocław
wroclaw_center = [51.117045, 17.001045]

# Inicjalizacja mapy
m = folium.Map(location=wroclaw_center, zoom_start=13)

# Dodanie markerów
for result in results_data['results']:
    lat = result['geocodes']['main']['latitude']
    lon = result['geocodes']['main']['longitude']
    name = result['name']
    categories = ', '.join([category['name'] for category in result['categories']])
    address = result['location']['formatted_address']

    popup_html = f"<b>{name}</b><br>{categories}<br>{address}"
    folium.Marker([lat, lon], popup=popup_html).add_to(m)

# Zapisz mapę do pliku HTML
m.save("viz/mapa_wroclawia.html")