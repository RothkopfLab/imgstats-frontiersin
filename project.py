if __name__ == "__main__":
    import argparse
    from projection import transform_samples

    parser = argparse.ArgumentParser(description='Transform images from projective plane to tangential plane.')
    parser.add_argument('directory', metavar='dir', type=str, help='directory of image samples')
    parser.add_argument('-s', '--size', type=int, default=128, help='size of output images')

    args = parser.parse_args()

    transform_samples(dataset=args.directory, patch_size=args.size)
