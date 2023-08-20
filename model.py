import torch
import torch.nn as nn
import requests

API_KEY = '995f53dfd607ffb87111125eb546731f:8b5b681984d32fcebb303c10d9dfa031'
PUBLIC_KEY_ID = 'c949f451-cf77-54ee-897c-91a7c4da3d62'
url = f'https://api.circle.com/v2/notifications/publicKey/%7BPUBLIC_KEY_ID%7D'


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
        self.User_Requested = False
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

    def get_circle_data(self):
        headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {API_KEY}',
        }
        if self.User_Requested:
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                print("Notifications received:")
                for notification in data['data']:
                    print(notification)
            else:
                print(f"Request failed with status code: {response.status_code}")
                print(response.text)
    
    def set_User_Requested(self, User_Requested):
        self.User_Requested = User_Requested






