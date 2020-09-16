# Client Server Data Streaming

1. Download or clone this repo

2. Extract the repo to your hard disk

3. Open Command Prompt and type the command **ipconfig**, note down the IPv4 address of the Wi-Fi adapter

4. In client.py **line 14**, modify the url to math your IP address leaving the port 8080 intact

4. Open **3** Anaconda Prompts and in each one change directory to the downloaded repository using cd

5. In the first Anaconda Prompt, navigate to **server** folder, run **python -m http.server** to initiate FTP server. Allow access to bypass firewall. **Tick public and private networks both** 

<img src="https://i.stack.imgur.com/VLdf5.png" width="500">

6. In the second Anaconda Prompt run **python steer_server.py** which will load tensorflow model and write steering angle to disk as an npy file

7. On Raspberry Pi, run **python client.py** which will download the npy file and print steering angles on terminal as well as visualize it on a steering wheel

