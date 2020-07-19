import os
import requests
import numpy as np

from enum import Enum
from queue import Empty
from multiprocessing import Pool, Manager, Queue
from typing import NoReturn, Dict, Any, Iterable, List, Union


class BatchMode(Enum):
    """
    `BatchMode` provides enumerate values to create multiple parameter combination. Let's say you have passed the
    following two parameters, i.e. `p1` and `p2`

    >>> p1 = [1, 2]
    >>> p2 = [3, 4]

    There are two `BatchMode` values available:

    - **`BatchMode.scan`** will create two parameter combining these values:

        - `{'p1': 1, 'p2': 3}`
        - `{'p1`: 2, 'p2`: 4}'

    - **`BatchMode.combine`** will create four as follows:

        - `{'p1': 1, 'p2': 3}`
        - `{'p1`: 2, 'p2`: 3}'
        - `{'p1': 1, 'p2': 4}`
        - `{'p1`: 2, 'p2`: 4}'
    """
    scan = 'scan'
    combine = 'combine'


class ezrequest:
    """
    `ezrequest` wraps around the famous `requests` package making it easier to use in certain cases.

    :param url: The url of the API to be called. For example, in `http://httpbin.org/get` the url is
                `http://httpbin.org`.
    :param path: The path or resource to be added to the url. For example, in `http://httpbin.org/get`
                 the url is `/get`.
    :param fixed_params: Parameters that needs to be added to the url everytime that it is called.
    :param \*\*kwargs: additional named values. Currently it is not used.

    :type url: str
    :type path: str
    :type fixed_params: Dict[str, Any]
    :type \*\*kwargs: Dict[str, Any]

    The followings are some examples showing how to use `ezrequest`:

    Example:
        **No Parameter at all**

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response = er.get() # will perform a GET call to the url: https://httpbin.org/get
        >>> type(response)
        <class 'requests.models.Response'>
        >>> import pprint
        >>> pretty_print = pprint.PrettyPrinter(indent=2)
        >>> pretty_print.pprint(response.json())
        { 'args': {},
          'headers': { 'Accept': '*/*',
                       'Accept-Encoding': 'gzip, deflate',
                       'Host': 'httpbin.org',
                       'User-Agent': 'python-requests/2.24.0',
                       'X-Amzn-Trace-Id': 'masked-output'},
          'origin': 'masked-output',
          'url': 'https://httpbin.org/get'}
        >>> response.url
        'https://httpbin.org/get'

    Example:
        **Passing different Parameters in each `GET` call**

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response = er.get(variable_params={'p1': '1', 'p2': '2'})
        >>> response.url
        'https://httpbin.org/get?p1=1&p2=2'
        >>> response = er.get(variable_params={'p1': '3', 'p2': '4'})
        >>> response.url
        'https://httpbin.org/get?p1=3&p2=4'

    Example:
        **Fixing some parameters across calls**

        Fixed Parameters are passed at the instantiation and during the initialization. The variable parameters are
        passed at each `GET` call. The fixed parameters are added to the variable parameters. Using the same object
        across multiple thread should be safe for different `variable_params`; however, changing `fixed_params` is not
        guaranteed to be thread-safe.

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get', fixed_params={'fixP': '42'})
        >>> response = er.get(variable_params={'p1': '1', 'p2': '2'})
        >>> response.url
        'https://httpbin.org/get?fixP=42&p1=1&p2=2'
        >>> response = er.get(variable_params={'p1': '3', 'p2': '4'})
        >>> response.url
        'https://httpbin.org/get?fixP=42&p1=3&p2=4'

    Example:
        **Batch Get Calls**

        Let's say you want to try two different parameters: (1) `{'p1': 1, 'p2': 3}` and (2) `{'p1': 2, 'p2': 4}`. One
        option is to do two different get call, one for each parameter. The other option is to use the `batch_get()` as
        follows:

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> param_list = [{'p1': 1, 'p2': 3},  {'p1': 2, 'p2': 4}]
        >>> response_list = er.batch_get(param_list)
        >>> for response in response_list:
        ...     print(response.url)
        ...
        https://httpbin.org/get?p1=1&p2=3
        https://httpbin.org/get?p1=2&p2=4

    Example:
        **Batch Get Calls in parallel**

        As previous example, you have multiple parameters, let's say just two in this case. And you want to call them
        all but this time using parallel. You can use the `batch_get()` in parallel by setting the `use_parallel` to
        `True` as follows:

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> param_list = [{'p1': 1, 'p2': 3},  {'p1': 2, 'p2': 4}]
        >>> response_list = er.batch_get(param_list, use_parallel=True)
        >>> for response in response_list:
        ...     print(response.url)
        ...
        https://httpbin.org/get?p1=1&p2=3
        https://httpbin.org/get?p1=2&p2=4

        **NOTE:** for small number of parameters as in this example, the parallel version will take even longer. The
        prallel version needs to spin-up new python processes, which would take some time. It only make sense once to
        use it if you have a lot of different values listed in the param_list.

        When setting `use_parallel` to `True`, by default, new processes are spawned and the pramaters are queued to be
        processed by these processes. The default number of processes depends on the value that `os.cpu_count()`
        returns. However, you can also manually define this number:

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> param_list = [{'p1': 1, 'p2': 3},  {'p1': 2, 'p2': 4}]
        >>> response_list = er.batch_get(param_list, use_parallel=True, pool=2)
        >>> for response in response_list:
        ...     print(response.url)
        ...
        https://httpbin.org/get?p1=1&p2=3
        https://httpbin.org/get?p1=2&p2=4

        You can also reuse the previously created process pool, to avoid the creation of new processed.

        >>> from multiprocessing import Pool
        >>> from ezrequest import BatchMode, ezrequest, prep_param_list
        >>> pool = Pool(processes=3)
        >>> # doing some other work with the pool
        ... # Now reusing the same pool
        ... p1=[1, 2, 3]
        >>> p2=[4, 5, 6]
        >>> p3=[7, 8, 9]
        >>> # using prep_param_list to quickly create a parameter list
        ... # refer to its documentation for more information
        ... param_list = prep_param_list(BatchMode.combine, {'p1': p1, 'p2': p2, 'p3': p3})
        >>> len(param_list)
        27
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response_list = er.batch_get(
        ...     param_list=param_list,
        ...     use_parallel=True,
        ...     pool=pool)
        >>> len(response_list)
        27

    Example:
        **Batch Get Calls by scanning parameters**

        Let's say you want to perform two get calls: (1) for `{'p1': 1, 'p2': 3}` and (2) for `{'p1': 2, 'p2': 4}`.
        one approach, of course is to do it one by one:

        >>> from ezrequest import ezrequest
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response = er.get(variable_params={'p1': '1', 'p2': '3'})
        >>> response.url
        'https://httpbin.org/get?p1=1&p2=3'
        >>> response = er.get(variable_params={'p1': '2', 'p2': '4'})
        >>> response.url
        'https://httpbin.org/get?p2=3&p2=4'

        Another approach is to use `ezrequest.batch_get_s`, as follows:

        >>> from ezrequest import ezrequest, BatchMode
        >>> er = ezrequest(url='https://httpbin.org', path='/get', fixed_params={'fixP': '42'})
        >>> response_list = er.batch_get_s(batch_mode=BatchMode.scan, p1=[1, 2], p2=[3, 4])
        >>> len(response_list)
        2
        >>> [str(response.url) for response in response_list]
        ['https://httpbin.org/get?fixP=42&p1=1&p2=3', 'https://httpbin.org/get?fixP=42&p1=2&p2=4']

        `ezrequest.batch_get_s` will return a list of responses for each parameter.

    Example:
        **Batch Get Calls by combining parameters**

        Let's say, again, you have two values for `p1 = [1, 2]` and two values for `p2 = [3, 4]` as in prior example.
        However, this time, instead of scanning through the list and creating a variable parmeter by matching each
        elements together, you want to try out all the combinations, i.e. all combination constructed by trying each
        values for `p1` for each values of `p2`. In this case, it would be 4 different combinations. Instead of
        manually constructing all these calls, you can again use the `ezrequest.batch_get_s`, however, this time, set
        `batch_mode` to `BatchMode.combine`:

        >>> from ezrequest import ezrequest, BatchMode
        >>> er = ezrequest(url='https://httpbin.org', path='/get', fixed_params={'fixP': '42'})
        >>> response_list = er.batch_get_s(batch_mode=BatchMode.combine, p1=[1, 2], p2=[3, 4])
        >>> len(response_list)
        4
        >>> for response in response_list:
        ...     print(response.url)
        ...
        https://httpbin.org/get?fixP=42&p1=1&p2=3
        https://httpbin.org/get?fixP=42&p1=2&p2=3
        https://httpbin.org/get?fixP=42&p1=1&p2=4
        https://httpbin.org/get?fixP=42&p1=2&p2=4

    Example:
        **Having more than 2 parameters**

        You can have more than two parameters and more than two values for each parameter. For example:

        >>> from ezrequest import ezrequest, BatchMode
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response_list = er.batch_get_s(
        ...     batch_mode=BatchMode.combine,
        ...     p1=[1, 2],
        ...     p2=[3, 4, 5],
        ...     p3=[6, 7, 8, 9]
        ... )
        >>> len(response_list)
        24

        There are 24 responses retrieved, because you had 2 values for `p1`, 3 values for `p2`, and 4 values for `p3`,
        hence 2 * 3 * 4 = 24.

    Example:
        **Batch Call in parallel**

        the `s` at the end of `batch_get_s()` means that it is a serial version of the batch call. This means, that all
        the GET calls would be performed serially, one after another. However, if you have a lot of combinations or
        variations to scan through, that could take some time. If possible to send these calls in parallel you can use
        `batch_get_p()` instead.

        For example, if you want to have 3 processes to perform the get calls simultanously, you would issue the
        following:

        >>> from ezrequest import ezrequest, BatchMode
        >>> er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response_list = er.batch_get_p(
        ...     pool=3,
        ...     batch_mode=BatchMode.combine,
        ...     p1=[1, 2, 3],
        ...     p2=[4, 5, 6],
        ...     p3=[7, 8, 9])
        >>> len(response_list)
        27

    Example:
        **Reusing a process pool**

        If you already have a pool of processes created somewhere in your code, you code use the same pool, instead of
        starting new processes:

        >>> from multiprocessing import Pool
        >>> pool = Pool(processes=3)
        >>> # doing some other work with the pool
        ... # Now reusing the same pool
        ... er = ezrequest(url='https://httpbin.org', path='/get')
        >>> response_list = er.batch_get_p(
        ...     pool=pool,
        ...     batch_mode=BatchMode.combine,
        ...     p1=[1, 2, 3],
        ...     p2=[4, 5, 6],
        ...     p3=[7, 8, 9])
        >>> len(response_list)
        27

    """
    def __init__(self, url: str, path: str, fixed_params: Dict[str, Any] = None, **kwargs) -> NoReturn:
        """
        Initializes an instance of ezrequest.

        :param url: The url of the API to be called. For example, in `http://httpbin.org/get` the url is
                `http://httpbin.org`.
        :param path: The path or resource to be added to the url. For example, in `http://httpbin.org/get`
                     the url is `/get`.
        :param fixed_params: Parameters that needs to be added to the url everytime that it is called.
        :param \*\*kwargs: additional named values. Currently it is not used.

        :type url: str
        :type path: str
        :type fixed_params: Dict[str, Any]
        :type \*\*kwargs: Dict[str, Any]
        """

        # print(f'[{os.getpid()}] initializing ezrequest ...')
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
        """
        :return: The base url, for example, in `http://httpbin.org/get` the url is `http://httpbin.org`.

        :rtype: str
        """
        return self._url

    @property
    def path(self) -> str:
        """
        :return: The path in url, for example, in `http://httpbin.org/get`, the path is `/get`.

        :rtype: str
        """
        return self._path

    @property
    def fixed_params(self) -> Dict[str, Any]:
        """
        :return: the fixed parameters provided at the time of instantiations. These parameters are passed in each call.
        :rtype: Dict[str, any]
        """
        return dict(self._fixed_params)

    @property
    def full_url(self) -> str:
        """
        :return: full url consisting of base url and path.

        :rtype: str
        """
        return self.url + self._path

    @property
    def session(self) -> requests.Session:
        """
        :return: Gives you access to the actual `requests.Session` instance that is used under the hood.

        :rtype: requests.Session
        """
        self._prep_session()
        return self._session

    def close(self) -> NoReturn:
        """
        closes the session. Consequent get calls, will result in creating a new session.
        """
        if self._session is not None:
            self.session.close()
            self._session = None

    def _process_parameter(self, in_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        allows modification or further processing of the provided parameters. This default implementation does nothing.
        you could override this method in child classes to change the behavior.

        :param in_params: input parameters that are processed

        :type in_params: Dict[str, Any]

        :return: Processed parameters

        :rtype: Dict[str, Any]
        """
        return in_params

    def get(self, variable_params: Dict[str, Any] = None) -> requests.models.Response:
        """
        :param variable_params: The parameters to be added when performing the REST call.

        :type variable_params: Dict[str, Any]

        :return: Returns a response received after performing a REST Get call.

        :rtype: requests.models.Response:
        """
        if variable_params is None:
            variable_params = {}

        return self.session.get(
            self.full_url,
            params=self._process_parameter(variable_params)
        )

    def batch_get(self,
                  param_list: Iterable[Dict[str, Any]],
                  use_parallel: bool = False,
                  pool: Union[int, Pool] = os.cpu_count()):
        """
        performs a REST GET call for each parameter that is provided in the parameter list. This is good if you have to
        make multiple calls. You could also perform the calls in parallel.
        :param param_list: An Iterable containing multiple parameters that needs to be evaluated for REST Get call.

        :type param_list: Iterable[Dict[str, Any]]

        :param use_parallel: Determines if to make the REST calls in parallel or sequentially. Default is False.

        :type use_parallel: bool

        :parma pool: either an integer specifying the number of processes to be created, or an already created pool
                     of processes. Default is number of cpu counts as returned by `os.cpu_count()`.

        :type pool: Union[int, Pool]

        :return: a list of requests responses

        :rtype: List[requests.models.Response]
        """

        response = []
        if use_parallel:
            processes_pool = Pool(processes=pool) if isinstance(pool, int) else pool
            param_queue = Manager().Queue()
            for param in param_list:
                param_queue.put(param)

            n_processes = int(processes_pool._processes)
            # print(f'[{os.getpid()}]: starting {n_processes} processes.')
            process_response = [None] * n_processes
            for idx in range(n_processes):
                process_response[idx] = processes_pool.apply_async(
                    ezrequest._batch_get_p_task,
                    (self.url, self.path, self.fixed_params, self._kwargs, param_queue)
                )

            for idx in range(n_processes):
                response.extend(process_response[idx].get())
        else:
            response.extend([self.get(param) for param in param_list])

        return response

    def batch_get_s(self,
                    batch_mode: Union[BatchMode, str] = BatchMode.scan,
                    **parameters_dict_list) -> List[requests.models.Response]:
        """
        Performs a batch get call. Refer to examples, for instruction on how to use.

        :param batch_mode: determines how the parameters must be constructed.

        :type batch_mode: BatchMode

        :param parameters_dict_list: a dictionary of various parameters with multiple values.

        :type parameters_dict_list: Dict[str, Iterable[Any]]

        :return: a list of requests responses

        :rtype: List[requests.models.Response]
        """
        param_list = prep_param_list(batch_mode, parameters_dict_list)
        return self.batch_get(param_list=param_list)

    def batch_get_p(self,
                    pool: Union[int, Pool] = os.cpu_count(),
                    batch_mode: Union[BatchMode, str] = BatchMode.scan,
                    **parameters_dict_list) -> List[requests.models.Response]:
        """
        The same as `batch_get_s` performs a batch of REST calls, however, in parallel.

        :parma pool: either an integer specifying the number of processes to be created, or an already created pool
                     of processes.

        :type pool: Union[int, Pool]

        :param batch_mode: determines how the parameters must be constructed.

        :type batch_mode: BatchMode

        :param parameters_dict_list: a dictionary of various parameters with multiple values.

        :type parameters_dict_list: Dict[str, Iterable[Any]]

        :return: a list of requests responses

        :rtype: List[requests.models.Response]
        """

        param_list = prep_param_list(batch_mode, parameters_dict_list)

        return self.batch_get(
            param_list=param_list,
            use_parallel=True,
            pool=pool
        )

    @staticmethod
    def _batch_get_p_task(url: str,
                          path: str,
                          fixed_params: Dict[str, Any],
                          kwargs: Dict[Any, Any],
                          param_queue: Queue) -> List[requests.models.Response]:
        # print(f'[{os.getpid()}] starting task ...')
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
                # print(f'[{os.getpid()}]: param: {str(param)}')
                response.append(er.get(param))
        except Empty:
            pass

        # print(f'[{os.getpid()}] ending task ...')

        return response

    def effective_url(self, params=None) -> str:
        if params is None:
            params = {}
        return (self.get(variable_params=self._process_parameter(params))).url

    def __enter__(self) -> 'ezrequest':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> NoReturn:
        self.session.close()

    def __del__(self) -> NoReturn:
        self.session.close()


def prep_param_list(
        batch_mode: Union[BatchMode, str],
        parameters_dict_list: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """
    Creates a list of parameters out of the multiple values that are provided for each parameter.

    :param batch_mode: defines how to create a parameter list. Poosible options or `BatchMode.scan` and
                       `BatchMode.combine`

    :type batch_mode: BatchMode

    :param parameters_dict_list: a dictionary that the keys are the parameter name and the values are iterable
                                 providing multiple values for each parameter.

    :type parameters_dict_list: Dict[str, Iterable[Any]]

    :return: A list of parameters created out of the provided parameters.

    :rtype: List[Dict[str, Any]]

    Example:
        **Creating parameters using `BatchMode.scan`**

        Let's say you have parameter `a` which has two values: `[1, 2]`, and parameter `b` which also has two values:
        `[3, 4]`. You are interseted to create a list of parameters which is constructed from each values of parameter
        `a` with the corresponding one in parameter `b`:

        >>> from ezrequest import BatchMode, prep_param_list
        >>> plist = prep_param_list(BatchMode.scan, {'a': [1, 2], 'b': [3, 4]})
        >>> print(plist)
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]

    Example:
        **Creating parameters by combining the values (trying each combination)**

        This is the same as before, except that you want to try all the combinations.

        >>> from ezrequest import BatchMode, prep_param_list
        >>> plist = prep_param_list(BatchMode.combine, {'a': [1, 2], 'b': [3, 4]})
        >>> for parameter in plist:
        ...     print(parameter)
        ...
        {'a': 1, 'b': 3}
        {'a': 2, 'b': 3}
        {'a': 1, 'b': 4}
        {'a': 2, 'b': 4}

    Example:
        **More than two parameters**

        If you have more than two variables, just list them.

        >>> from ezrequest import BatchMode, prep_param_list
        >>> plist = prep_param_list(BatchMode.combine, {'a': [1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8, 9]})
        >>> for parameter in plist:
        ...     print(parameter)
        ...
        {'a': 1, 'b': 3, 'c': 6}
        {'a': 1, 'b': 3, 'c': 7}
        {'a': 1, 'b': 3, 'c': 8}
        {'a': 1, 'b': 3, 'c': 9}
        {'a': 2, 'b': 3, 'c': 6}
        {'a': 2, 'b': 3, 'c': 7}
        {'a': 2, 'b': 3, 'c': 8}
        {'a': 2, 'b': 3, 'c': 9}
        {'a': 1, 'b': 4, 'c': 6}
        {'a': 1, 'b': 4, 'c': 7}
        {'a': 1, 'b': 4, 'c': 8}
        {'a': 1, 'b': 4, 'c': 9}
        {'a': 2, 'b': 4, 'c': 6}
        {'a': 2, 'b': 4, 'c': 7}
        {'a': 2, 'b': 4, 'c': 8}
        {'a': 2, 'b': 4, 'c': 9}
        {'a': 1, 'b': 5, 'c': 6}
        {'a': 1, 'b': 5, 'c': 7}
        {'a': 1, 'b': 5, 'c': 8}
        {'a': 1, 'b': 5, 'c': 9}
        {'a': 2, 'b': 5, 'c': 6}
        {'a': 2, 'b': 5, 'c': 7}
        {'a': 2, 'b': 5, 'c': 8}
        {'a': 2, 'b': 5, 'c': 9}

    Example:
        **Using `BatchMode.scan` with different number of values for each parameter**

        Note that if you have different number of values (or options) for each parameter, and decide to use the
        `BatchMode.scan` instead of `BatchMode.combine` the shortest iterable will decide how many parameter is created.

        >>> from ezrequest import BatchMode, prep_param_list
        >>> plist = prep_param_list(BatchMode.scan, {'a': [1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8, 9]})
        >>> for parameter in plist:
        ...     print(parameter)
        ...
        {'a': 1, 'b': 3, 'c': 6}
        {'a': 2, 'b': 4, 'c': 7}

        As you can see only the first two values for each list is used. Value 5 for `b` is never used, and valu 8 and 9
        for `c` are also never used.
    """

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
