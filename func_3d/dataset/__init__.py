
from .data_utils import DatasetTest
from torch.utils.data import DataLoader


def get_dataloader_test(args, organ):

    test_dataset = DatasetTest(args, args.dataset, args.data_path, organ, args.sup_num, args.sup_vol_num, transform=None, mode='Test',
                               prompt=args.test_prompt)
    nice_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return nice_test_loader


