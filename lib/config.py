
class config:
    def __init__(self,**kwargs):
        self.iter_epoch=[0,100]
        self.iter_beta=[0.2,0.4]
        self.iter_size=[1,1]
        self.iter_drop=[0.1,0.3]
        for keys in kwargs.keys():
            setattr(self,keys,kwargs[keys])
