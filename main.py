from modules.interface import create_interface

iface = create_interface()
iface.launch(server_name="0.0.0.0", server_port=8080)