import pyftpdlib.authorizers
import pyftpdlib.handlers
import pyftpdlib.servers

#establish file path and user login details
authorizer = pyftpdlib.authorizers.DummyAuthorizer()
authorizer.add_user('nao', 'nao', 'C:\\3A\\projet_mos\\images', perm='elradfmw')

#handles individual connections
handler = pyftpdlib.handlers.FTPHandler
handler.authorizer = authorizer

#start up server
server = pyftpdlib.servers.FTPServer(("192.168.1.3", 8000), handler)
server.serve_forever()