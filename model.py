import torch
import torch.nn as nn


#(kernel_size, filters, stride, padding) if list: (kernel_size, filters, stride, padding,repeats)
#"M" is maxpool 
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1)) ##start_dim = 1 as we don't want to flatten number of examples?????
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels,
                            out_channels=x[1],
                            kernel_size=x[0],
                            stride=x[2],
                            padding=x[3]
                            )
                        ]
                in_channels = x[1]
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                number_of_repeats = x[2]
                
                for _ in range(number_of_repeats):
                    layers += [
                        CNNBlock(
                        in_channels,
                        conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                        conv1[1],
                        conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
        
        return nn.Sequential(*layers)
    
    #https://miro.medium.com/v2/resize:fit:4800/format:webp/1*YG6heD55fEmZeUKRSlsqlA.png 1-5 bounding box(confidence,x,y,w,h), 
    #                                                                                    6 - 10 box(same params),
    #                                                                                    11-30 class probabilities
    #                                                                                    why 7x7x30?:
    #                                                                                               start image was divided into 7x7 parts(that's what YOLO algorithm is about) and each cell(of 7x7) has vector(lenght=30) that was described above
    #https://miro.medium.com/v2/resize:fit:1400/format:webp/1*q5feieizWKYq7dpWjYvCOw.png
                
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )
    
