from    Test.Evaluate import ShowSamples
from    Train.Init import DataInit, ModelInit

def train():

    hyper_dict, model, optimizer, lossFn = ModelInit(14)
    DI = DataInit(path, hyper_dict['bz'])
    test_loader, train_loader = DI.Reality()

    for epoch in range(hyper_dict['ep']):
        for iter, (img, GTkp, centermap) in enumerate(train_loader):
            model.train()
            img, GTkp, centermap = img.to(hyper_dict['dv']), GTkp.to(hyper_dict['dv']), centermap.to(hyper_dict['dv'])
            print(iter, img.shape, GTkp.shape, centermap.shape)
            heat_maps = model(img, centermap)
            temp_loss = []
            for map in heat_maps:
                temp_loss.append(lossFn(map, GTkp))
            loss = sum(temp_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ShowSamples(model, hyper_dict['dv'])
        print(f'No.{epoch} has been done!')




if __name__ == '__main__':
    path = r"E:\9_A_PhD\DataSet\Leeds_Sport_Pose\DataSet"
    train()