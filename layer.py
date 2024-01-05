#basic layer framework

class Layer():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input):
        return

    def backward(self,output_gradient,learning_rate):
        return
