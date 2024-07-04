import json

class NodeRegistry:
    """
    A class to manage the configuration of the nodes in the network.
    
    The configuration of the nodes is stored in a json file. 
    The class provides methods to access the configuration of the nodes.
    """
    
    def __init__(self, config_file):
        """
        Input: config_file - a json file containing the configuration of the nodes
        
        The configuration file should be a json file with the following format:
        {
            "node1": {
                "ip": ""
                "path": "/path/to/node1",
                "worker_name": "worker1"
            },
            "node2": {
                "ip": ""
                "path": "/path/to/node2",
                "worker_name": "worker2"
            }
        }
        
        The keys are the node ids, and the values are dictionaries containing the ip, path, and worker_name of the node.
        """
        with open(config_file) as f:
            self.config = json.load(f)
        
    def get_all_node_ids(self):
        """
        Output: a list of all the node ids
        
        The node ids are the keys of the configuration dictionary.
        """
        return self.config.keys()
    
    def get_node_ip(self, node_id):
        """
        Input: node_id - the id of the node
        Output: the ip of the node
        
        The ip is a string.
        """
        return self.config[node_id]['ip']
    
    def get_node_path(self, node_id):
        """
        Input: node_id - the id of the node
        Output: the path of the node
        
        The path is a string.
        """
        return self.config[node_id]['path']
    
    def get_node_worker_name(self, node_id):
        """
        Input: node_id - the id of the node
        Output: the worker_name of the node
        
        The worker_name is a string.
        """
        return self.config[node_id]['worker_name']
    
    def get_node_password(self, node_id):
        """
        Input: node_id - the id of the node
        Output: the password of the node
        
        The password is a string.
        """
        return self.config[node_id]['password']
 
