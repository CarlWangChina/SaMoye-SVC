import os
import boto3
from dataclasses import dataclass
from pathlib import Path
from .hparams import get_hparams
from .logger import get_logger

logger = get_logger(__name__)


# s3 实例
@dataclass
class S3:
    hparams_s3: dict

    def _client(self):
        return boto3.client(
            "s3",
            region_name=self.hparams_s3["region_name"],
            aws_access_key_id=self.hparams_s3["aws_access_key_id"],
            aws_secret_access_key=self.hparams_s3["aws_secret_access_key"],
        )

    def upload_files(self, bucket_name, path_local, path_s3):
        """
        上传（重复上传会覆盖同名文件）
        :param path_local: 本地路径
        :param path_s3: s3路径
        """
        logger.info(f"Start upload files.")

        if not self.upload_single_file(bucket_name, path_local, path_s3):
            logger.info(f"Upload files failed.")

        logger.info(f"Upload files successful.")

    def upload_single_file(self, bucket_name, src_local_path, dest_s3_path):
        """
        上传单个文件
        :param src_local_path:
        :param dest_s3_path:
        :return:
        """
        try:
            with open(src_local_path, "rb") as f:
                self._client().upload_fileobj(f, bucket_name, dest_s3_path)
        except Exception as e:
            logger.info(
                f"Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}"
            )
            return False
        logger.info(
            f"Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}"
        )
        return True

    def download_single_file(self, bucket_name, path_s3, path_local):
        """
        下载
        :param path_s3:
        :param path_local:
        :return:
        """
        retry = 0
        result = 0
        while retry < 3:  # 下载异常尝试3次
            logger.info(
                f"Start downloading files. | path_s3: {path_s3} | path_local: {path_local}"
            )
            try:
                self._client().download_file(bucket_name, path_s3, path_local)
                logger.info(f"Downloading completed. | size: {path_local}")
                result = 1
                break  # 下载完成后退出重试
            except Exception as e:
                logger.info(f"Download zip failed. | Exception: {e}")
                retry += 1

        if retry >= 3:
            logger.info(f"Download zip failed after max retry.")
            return result

    def delete_s3_zip(self, bucket_name, path_s3, file_name=""):
        """
        删除
        :param path_s3:
        :param file_name:
        :return:
        """
        try:
            # copy
            # copy_source = {'Bucket': bucket_name, 'Key': path_s3}
            # self._client().copy_object(CopySource=copy_source, Bucket=BUCKET_NAME, Key='is-zips-cache/' + file_name)
            self._client().delete_object(Bucket=bucket_name, Key=path_s3)
        except Exception as e:
            logger.info(f"Delete s3 file failed. | Exception: {e}")
        logger.info(f"Delete s3 file Successful. | path_s3 = {path_s3}")

    def batch_delete_s3(self, bucket_name, delete_key_list):
        """
        批量删除
        :param delete_key_list: [
                    {'Key': "test-01/虎式03的副本.jpeg"},
                    {'Key': "test-01/tank001.png"},
                ]
        :return:
        """
        try:
            res = self._client().delete_objects(
                Bucket=bucket_name, Delete={"Objects": delete_key_list}
            )
        except Exception as e:
            logger.info(f"Batch delete file failed. | Excepthon: {e}")
        logger.info(f"Batch delete file success. ")

    def get_files_list(self, bucket_name, Prefix=None):
        """
        查询
        :param start_after:
        :return:
        """
        logger.info(f"Start getting files from self._client().")
        try:
            if Prefix is not None:
                all_obj = self._client().list_objects_v2(
                    Bucket=bucket_name, Prefix=Prefix
                )

                # 获取某个对象的head信息
                # obj = self._client().head_object(Bucket=BUCKET_NAME, Key=Prefix)
                # logger.info(f"obj = {obj}")
            else:
                all_obj = self._client().list_objects_v2(Bucket=bucket_name)

        except Exception as e:
            logger.info(f"Get files list failed. | Exception: {e}")
            return

        contents = all_obj.get("Contents")
        if not contents:
            return

        file_name_list = []
        for zip_obj in contents:
            # logger.info(f"zip_obj = {zip_obj}")
            file_size = round(zip_obj["Size"] / 1024 / 1024, 3)  # 大小
            # logger.info(f"file_path = {zip_obj['Key']}")
            # logger.info(f"LastModified = {zip_obj['LastModified']}")
            # logger.info(f"file_size = {file_size} Mb")
            # zip_name = zip_obj['Key'][len(start_after):]
            zip_name = zip_obj["Key"]

            file_name_list.append(zip_name)

        logger.info(f"Get file list successful.")

        return file_name_list

    def upload_dir(self, source_dir: str, s3_dir: str, bucket_name: str):
        """ """
        pdir = Path(source_dir)
        flist = os.listdir(str(pdir))
        for i in flist:
            path_local = str(pdir / i)
            path_s3 = str(Path(s3_dir) / pdir.name / i)
            self.upload_single_file(
                bucket_name=bucket_name, src_local_path=path_local, dest_s3_path=path_s3
            )

    def download_dir(self, s3_dir: str, save_path: str, bucket_name: str):
        flist = self.get_files_list(bucket_name=bucket_name, Prefix=s3_dir)
        save_dir = Path(s3_dir).name
        save_pdir = Path(save_path) / save_dir
        if not save_pdir.exists():
            os.mkdir(str(save_pdir))
        for i in flist:
            fname = Path(i).name
            path_local = save_pdir / fname
            self.download_single_file(
                bucket_name=bucket_name, path_s3=i, path_local=path_local
            )
        # logger.info(flist)

    def file_exist(self, bucket_name: str, file_name: str):
        import boto3

        # 连接到AWS S3
        s3 = boto3.client("s3")

        # 检查文件是否存在
        bucket_name = "your_bucket_name"
        file_key = "your_file_key"

        try:
            info = self._client().head_object(bucket_name=bucket_name, key=file_key)
            logger.info(info)
            logger.info(f"文件 {file_key} 存在于存储桶 {bucket_name} 中")
        except Exception as e:
            logger.info(f"文件 {file_key} 不存在于存储桶 {bucket_name} 中")


s3 = None


def get_s3():
    global s3
    if s3 is None:
        s3_hparams = get_hparams()["s3"]
        s3 = S3(s3_hparams)
    return s3
