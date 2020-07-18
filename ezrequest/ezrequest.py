import os
import requests
import numpy as np

from enum import Enum
from queue import Empty
from multiprocessing import Pool, Manager, Queue
from typing import NoReturn, Dict, Any, Iterable, List, Union


class BatchMode(Enum):
    scan = 'scan'
    combine = 'combine'


class ezrequest:
    def __init__(self, url: str, path: str, fixed_params: Dict[str, Any] = None, **kwargs) -> NoReturn:
        print(f'[{os.getpid()}] initializing ezrequest ...')
        if fixed_params is None:
            fixed_params = {}
        self._url = url
        self._path = path
        self._fixed_params = fixed_params
        self._kwargs = kwargs
        self._session = None

    def _prep_session(self, force: bool = False) -> NoReturn:
        if (self._session is None) or force:
            self._session = requests.Session()
            self._session.params = self._fixed_params

    @property
    def url(self) -> str:
        return self._url

    @property
    def path(self) -> str:
        return self._path

    @property
    def fixed_params(self) -> Dict[str, Any]:
        return dict(self._fixed_params)

    @property
    def full_url(self) -> str:
        return self.url + self._path

    @property
    def session(self) -> requests.Session:
        self._prep_session()
        return self._session

    def close(self) -> NoReturn:
        self.session.close()

    def get(self, variable_params=None) -> requests.models.Response:
        if variable_params is None:
            variable_params = {}
        return self.session.get(self.full_url, params=variable_params)

    def batch_get_s(self,
                    batch_mode: Union[BatchMode, str] = BatchMode.scan,
                    **parameters_dict_list) -> List[requests.models.Response]:
        param_list = self.prep_param_list(batch_mode, parameters_dict_list)
        return [self.get(param) for param in param_list]

    def batch_get_p(self,
                    pool: Union[int, Pool] = os.cpu_count(),
                    batch_mode: Union[BatchMode, str] = BatchMode.scan,
                    **parameters_dict_list) -> List[requests.models.Response]:
        processes_pool = Pool(processes=pool) if isinstance(pool, int) else pool
        param_queue = Manager().Queue()
        for param in self.prep_param_list(batch_mode, parameters_dict_list):
            param_queue.put(param)

        n_processes = int(processes_pool._processes)
        print(f'[{os.getpid()}]: starting {n_processes} processes.')
        process_response = [None] * n_processes
        for idx in range(n_processes):
            process_response[idx] = processes_pool.apply_async(
                ezrequest._batch_get_p_task,
                (self.url, self.path, self.fixed_params, self._kwargs, param_queue)
            )

        response = []
        for idx in range(n_processes):
            response.extend(process_response[idx].get())

        return response

    @staticmethod
    def _batch_get_p_task(url: str,
                          path: str,
                          fixed_params: Dict[str, Any],
                          kwargs: Dict[Any, Any],
                          param_queue: Queue) -> List[requests.models.Response]:
        print(f'[{os.getpid()}] starting task ...')
        er = ezrequest(
            url=url,
            path=path,
            fixed_params=fixed_params,
            **kwargs
        )
        response = []
        try:
            while param_queue.qsize() > 0:  # still this could cause trouble; Hence, needs to be wrapped in try/catch
                param = param_queue.get(False)
                print(f'[{os.getpid()}]: param: {str(param)}')
                response.append(er.get(param))
        except Empty:
            pass

        print(f'[{os.getpid()}] ending task ...')

        return response

    def effective_url(self, params=None) -> str:
        if params is None:
            params = {}
        return (self.get(variable_params=params)).url

    def __enter__(self) -> 'ezrequest':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> NoReturn:
        self.session.close()

    def __del__(self) -> NoReturn:
        self.session.close()

    @staticmethod
    def prep_param_list(
            batch_mode: Union[BatchMode, str],
            parameters_dict_list: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
        batch_mode_enum = BatchMode(batch_mode)

        if batch_mode_enum == BatchMode.scan:
            # Checking All the parameters have the same
            return [
                dict(zip(parameters_dict_list.keys(), zipped_value))
                for zipped_value in zip(*parameters_dict_list.values())
            ]

        if batch_mode_enum == BatchMode.combine:
            return [
                dict(zip(parameters_dict_list.keys(), zipped_value))
                for zipped_value in zip(
                    *[
                        e.flatten()
                        for e in np.meshgrid(*parameters_dict_list.values())
                    ]
                )
            ]
