from multiprocessing import Pool
from unittest import TestCase

from ezrequest import ezrequest, BatchMode, prep_param_list


class Testezrequest(TestCase):
    def test_get_01(self):
        ezr = ezrequest(
            url='https://httpbin.org',
            path='/get')

        self.assertDictEqual({}, ezr.get().json()['args'])
        ezr.close()

    def test_with_fixedparams_get_01(self):
        fixed_params = {
            'k1': '1',
            'k2': '2'
        }
        ezr = ezrequest(
            url='https://httpbin.org',
            path='/get',
            fixed_params=fixed_params
        )

        self.assertDictEqual(fixed_params, ezr.get().json()['args'])
        ezr.close()

    def test_with_fixedparams_get_02(self):
        fixed_params = {
            'k1': '1',
            'k2': '2'
        }
        ezr = ezrequest(
            url='https://httpbin.org',
            path='/get',
            fixed_params=fixed_params
        )

        expected = {
            **fixed_params,
            'k3': '3',
            'k4': '4'
        }
        self.assertDictEqual(
            expected,
            ezr.get(
                {
                    'k3': '3',
                    'k4': '4'
                }
            ).json()['args']
        )
        ezr.close()

    def test_with_fixedparams_get_03(self):
        fixed_params = {
            'k1': '1',
            'k2': '2'
        }
        ezr = ezrequest(
            url='https://httpbin.org',
            path='/get',
            fixed_params=fixed_params
        )

        expected1 = {
            **fixed_params,
            'k3': '3',
            'k4': '4'
        }
        self.assertDictEqual(
            expected1,
            ezr.get(
                {
                    'k3': '3',
                    'k4': '4'
                }
            ).json()['args']
        )

        expected2 = {
            **fixed_params,
            'k3': '333',
            'k4': '4444'
        }
        self.assertDictEqual(
            expected2,
            ezr.get(
                {
                    'k3': '333',
                    'k4': '4444'
                }
            ).json()['args']
        )
        ezr.close()

    def test_with_fixedparams_get_04(self):
        fixed_params = {
            'k1': '1',
            'k2': '2'
        }

        expected = {
            **fixed_params,
            'k3': '3',
            'k4': '4'
        }

        with ezrequest(
            url='https://httpbin.org',
            path='/get',
            fixed_params=fixed_params
        ) as ezr:
            self.assertDictEqual(
                expected,
                ezr.get(
                    {
                        'k3': '3',
                        'k4': '4'
                    }
                ).json()['args']
            )

    def test_prep_param_list_01(self):
        plist = prep_param_list(BatchMode.scan, {'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(2, len(plist))
        self.assertTrue({'a': 1, 'b': 3}, plist[0])
        self.assertTrue({'a': 2, 'b': 4}, plist[1])

    def test_prep_param_list_02(self):
        plist = prep_param_list(BatchMode.combine, {'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(4, len(plist))
        self.assertDictEqual({'a': 1, 'b': 3}, plist[0])
        self.assertDictEqual({'a': 2, 'b': 3}, plist[1])
        self.assertDictEqual({'a': 1, 'b': 4}, plist[2])
        self.assertDictEqual({'a': 2, 'b': 4}, plist[3])

    def test_prep_param_list_03(self):
        plist = prep_param_list(BatchMode.scan, {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        self.assertEqual(2, len(plist))
        self.assertDictEqual({'a': 1, 'b': 3, 'c': 5}, plist[0])
        self.assertDictEqual({'a': 2, 'b': 4, 'c': 6}, plist[1])

    def test_prep_param_list_04(self):
        plist = prep_param_list(BatchMode.combine, {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        self.assertEqual(8, len(plist))
        self.assertDictEqual({'a': 1, 'b': 3, 'c': 5}, plist[0])
        self.assertDictEqual({'a': 1, 'b': 3, 'c': 6}, plist[1])
        self.assertDictEqual({'a': 2, 'b': 3, 'c': 5}, plist[2])
        self.assertDictEqual({'a': 2, 'b': 3, 'c': 6}, plist[3])
        self.assertDictEqual({'a': 1, 'b': 4, 'c': 5}, plist[4])
        self.assertDictEqual({'a': 1, 'b': 4, 'c': 6}, plist[5])
        self.assertDictEqual({'a': 2, 'b': 4, 'c': 5}, plist[6])
        self.assertDictEqual({'a': 2, 'b': 4, 'c': 6}, plist[7])

    def test_batch_get_s_01(self):
        e = ezrequest(url='https://httpbin.org', path='/get')

        response_list = e.batch_get_s(BatchMode.scan, a=[1, 2], b=[3, 4])
        self.assertEqual(2, len(response_list))
        self.assertDictEqual({'a': '1', 'b': '3'}, response_list[0].json()['args'])
        self.assertDictEqual({'a': '2', 'b': '4'}, response_list[1].json()['args'])

    def test_batch_get_s_02(self):
        e = ezrequest(url='https://httpbin.org', path='/get')

        response_list = e.batch_get_s(BatchMode.combine, a=[1, 2], b=[3, 4])
        self.assertEqual(4, len(response_list))
        self.assertDictEqual({'a': '1', 'b': '3'}, response_list[0].json()['args'])
        self.assertDictEqual({'a': '2', 'b': '3'}, response_list[1].json()['args'])
        self.assertDictEqual({'a': '1', 'b': '4'}, response_list[2].json()['args'])
        self.assertDictEqual({'a': '2', 'b': '4'}, response_list[3].json()['args'])

    def test_batch_get_p_01(self):
        er = ezrequest(url='https://httpbin.org', path='/get')

        response_list = [
            r.json()['args']
            for r in er.batch_get_p(3, BatchMode.scan, a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])
        ]
        self.assertEqual(3, len(response_list))
        self.assertTrue({'a': '1', 'b': '4', 'c': '7'} in response_list)
        self.assertTrue({'a': '2', 'b': '5', 'c': '8'} in response_list)
        self.assertTrue({'a': '3', 'b': '6', 'c': '9'} in response_list)

    def test_batch_get_p_02(self):
        er = ezrequest(url='https://httpbin.org', path='/get')

        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        response_list = [
            r.json()['args']
            for r in er.batch_get_p(3, BatchMode.combine, a=a, b=b, c=c)
        ]

        self.assertEqual(27, len(response_list))
        for a_e in a:
            for b_e in b:
                for c_e in c:
                    self.assertTrue({'a': str(a_e), 'b': str(b_e), 'c': str(c_e)} in response_list)


    def test_batch_get_p_03(self):
        er = ezrequest(url='https://httpbin.org', path='/get')

        pool = Pool(processes=3)
        response_list = [
            r.json()['args']
            for r in er.batch_get_p(pool, BatchMode.scan, a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])
        ]
        self.assertEqual(3, len(response_list))
        self.assertTrue({'a': '1', 'b': '4', 'c': '7'} in response_list)
        self.assertTrue({'a': '2', 'b': '5', 'c': '8'} in response_list)
        self.assertTrue({'a': '3', 'b': '6', 'c': '9'} in response_list)

    def test_batch_get_p_04(self):
        er = ezrequest(url='https://httpbin.org', path='/get')

        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        pool = Pool(processes=3)
        response_list = [
            r.json()['args']
            for r in er.batch_get_p(pool, BatchMode.combine, a=a, b=b, c=c)
        ]

        self.assertEqual(27, len(response_list))
        for a_e in a:
            for b_e in b:
                for c_e in c:
                    self.assertTrue({'a': str(a_e), 'b': str(b_e), 'c': str(c_e)} in response_list)