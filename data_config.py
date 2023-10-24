
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            #self.root_dir = '/media/lidan/ssd2/CDData/LEVIR-CD256/'
            self.root_dir = 'LEVIR'
        elif data_name == 'LEVIR-256':
            self.root_dir = 'LEVIR-CD256'
        elif data_name == 'test_LEVIR':
            self.root_dir = 'test_LEVIR'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = 'DSIFN'
        elif data_name == 'WHU256':
            self.label_transform = "norm"
            self.root_dir = 'WHU-256'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = 'WHU'
        elif data_name == 'CLCD':
            self.label_transform = "norm"
            self.root_dir = 'CLCD'
        elif data_name == 'CDD':
            self.root_dir = 'CDD'
        elif data_name == 'GZ':
            self.root_dir = 'GZ-CD'
        elif data_name == 'test':
            self.root_dir = 'test'
        elif data_name == 'test_WHU':
            self.root_dir = 'test_WHU'
        elif data_name == 'test_GZ':
            self.root_dir = 'test_GZ'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

