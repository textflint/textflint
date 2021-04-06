import unittest

from textflint.common.utils.list_op import *


class TestClass:
    pass


class TestListOp(unittest.TestCase):
    def test_normalize_scope(self):
        self.assertEqual([], normalize_scope([]))
        for i in range(100):
            a = random.randint(0, 1e8)
            b = random.randint(1e9, 1e10)
            self.assertEqual([a, b], normalize_scope([a, b]))
            self.assertEqual([a, b], normalize_scope((a, b)))
            self.assertEqual([a, a + 1], normalize_scope(a))
            self.assertEqual([a, b], normalize_scope(slice(a, b)))

    def test_replace_at_scopes(self):
        self.assertEqual([1, 2, 3, 4], replace_at_scopes([2, 2, 3, 4],
                                                         [[0, 1]], [[1]]))
        self.assertEqual([1, 2, 3, 4], replace_at_scopes(
            [2, 2, 3, 5], [[0, 1], [3, 4]], [[1], [4]]))
        self.assertEqual([1, 2, 3, 4], replace_at_scopes(
            [4, 2, 3, 1], [[0, 4]], [[1, 2, 3, 4]]))

        # Test successive replacement of individual elements
        num = random.randint(100, 1000)
        test_list = [random.randint(0, int(1e9)) for i in range(num)]
        test_pos = [[i, i + 1] for i in range(num)]
        random.shuffle(test_pos)
        self.assertEqual(list(range(num)), replace_at_scopes(
            test_list, test_pos, [[i[0]] for i in test_pos]))
        self.assertEqual(list(range(num)), replace_at_scopes(
            test_list, test_pos, [i[0] for i in test_pos]))

        # The test replaces multiple elements in succession
        l = random.randint(1, num - 2)
        test_pos = random.sample(list(range(1, num)), l)
        test_pos += [num, 0]
        test_pos.sort()
        test_pos = [[test_pos[i], test_pos[i + 1]] for i in range(l + 1)]
        random.shuffle(test_pos)
        self.assertEqual(list(range(num)), replace_at_scopes(
            test_list, test_pos, [[j for j in range(i[0], i[1])] for i in test_pos]))

        l = random.randint(2, 99)
        num = num - num % l
        test_list = test_list[:num]
        test_pos = [[i * l, i * l + l] for i in range(int(num / l))]
        random.shuffle(test_pos)
        self.assertEqual(list(range(num)), replace_at_scopes(
            test_list, test_pos, [[i[0] + j for j in range(l)] for i in test_pos]))

    def test_replace_at_scope(self):
        self.assertEqual([1, 2, 3, 4],
                         replace_at_scope([2, 2, 3, 4], [0, 1], [1]))
        self.assertEqual([1, 2, 3, 4],
                         replace_at_scope([1, 2, 3, 5], [3, 4], [4]))
        self.assertEqual([1, 2, 3, 4],
                         replace_at_scope([4, 2, 3, 1], [0, 4], [1, 2, 3, 4]))

        for i in range(50):
            a = random.randint(1, 500)
            b = random.randint(501, 999)
            test_list = list(range(1000))
            change = test_list[a:b]
            random.shuffle(change)
            test_list = test_list[:a] + change + test_list[b:]
            self.assertEqual(
                list(range(1000)),
                replace_at_scope(test_list, [a, b], list(range(a, b))))

    def test_delete_at_scopes(self):
        self.assertEqual([2, 3, 4], delete_at_scopes([1, 2, 3, 4], [0]))
        self.assertEqual([4], delete_at_scopes([1, 2, 3, 4], [0, 1, 2]))
        self.assertEqual([1, 2, 3], delete_at_scopes([1, 2, 3, 4], [3]))
        self.assertEqual([1, 4], delete_at_scopes([1, 2, 3, 4], [[1, 3]]))

        # Test successive replacement of individual elements
        num = random.randint(100, 1000)
        test_list = [random.randint(0, int(1e9)) for i in range(num)]
        test_pos = [[i, i + 1] for i in range(num)]
        random.shuffle(test_pos)
        self.assertEqual([], delete_at_scopes(test_list, [i for i in test_pos]))

        # Test replaces multiple elements in succession
        l = random.randint(1, num - 2)
        test_pos = random.sample(list(range(1, num)), l)
        test_pos += [num, 0]
        test_pos.sort()
        test_pos = [[test_pos[i], test_pos[i + 1]] for i in range(l + 1)]
        random.shuffle(test_pos)
        self.assertEqual([], delete_at_scopes(test_list, test_pos))

        l = random.randint(2, 99)
        num = num - num % l
        test_list = test_list[:num]
        test_pos = [[i * l, i * l + l] for i in range(int(num / l))]
        self.assertEqual([], delete_at_scopes(test_list, test_pos))

    def test_delete_at_scope(self):
        self.assertEqual([2, 3, 4], delete_at_scope([2, 2, 3, 4], 0))
        self.assertEqual([1, 2, 3], delete_at_scope([1, 2, 3, 5], [3, 4]))
        self.assertEqual([4, 1], delete_at_scope([4, 2, 3, 1], [1, 3]))

        num = random.randint(100, 1000)
        test_list = [random.randint(0, int(1e9)) for i in range(num)]
        l = random.randint(1, num - 2)
        test_pos = random.sample(list(range(1, num)), l)
        test_pos += [num, 0]
        test_pos.sort()
        test_pos = [[test_pos[i], test_pos[i + 1]] for i in range(l + 1)]
        for i in range(l):
            test_list = delete_at_scope(test_list, test_pos[l - i])
        self.assertEqual([], delete_at_scope(test_list, test_pos[0]))

    def test_insert_before_indices(self):
        self.assertEqual(
            [0, 1, 2, 3, 4],
            insert_before_indices([1, 2, 3, 4], [0], [0]))
        self.assertEqual(
            [1, 2, 3, 0, 1, 2, 4],
            insert_before_indices([1, 2, 3, 4], [3], [[0, 1, 2]]))
        self.assertEqual(
            [1, 2, 3, 5, 4],
            insert_before_indices([1, 2, 3, 4], [3], [5]))

        # Test successive replacement of individual elements
        num = random.randint(200, 1000)
        n = random.randint(20, 99)
        pos = random.sample(list(range(1, num - 1)), n)
        pos += [0, num]
        pos.sort()
        n += 2

        if n % 2 == 0:
            pos = pos[:-1]
            n -= 1
            pos[n - 1] = num
        test_pos = [[pos[i * 2], pos[i * 2 + 1]] for i in range(int(n / 2))]
        test_list = []
        origin = list(range(num))

        for i in range(int(n / 2)):
            test_list += origin[pos[2 * i + 1]:pos[2 * i + 2]]
        random.shuffle(test_pos)
        self.assertEqual(
            origin,
            insert_before_indices(
                test_list,
                [test_list.index(item[1]) for item in test_pos],
                [list(range(item[0], item[1])) for item in test_pos]))

    def test_insert_before_index(self):
        self.assertEqual(
            [0, 1, 2, 3, 4],
            insert_before_index([1, 2, 3, 4], 0, [0]))
        self.assertEqual(
            [0, 1, 2, 3, 4],
            insert_before_index([1, 2, 3, 4], 0, 0))
        self.assertEqual(
            [1, 2, 3, 0, 1, 2, 4],
            insert_before_index([1, 2, 3, 4], 3, [0, 1, 2]))
        self.assertEqual(
            [1, 2, 3, 5, 4], insert_before_index([1, 2, 3, 4], 3, [5]))

        # Test successive replacement of individual elements
        num = random.randint(200, 1000)
        n = random.randint(20, 99)
        pos = random.sample(list(range(1, num - 1)), n)
        pos += [0, num]
        pos.sort()
        n += 2

        if n % 2 == 0:
            pos = pos[:-1]
            n -= 1
            pos[n - 1] = num
        test_pos = [[pos[i * 2], pos[i * 2 + 1]] for i in range(int(n / 2))]
        test_list = []
        origin = list(range(num))

        for i in range(int(n / 2)):
            test_list += origin[pos[2 * i + 1]:pos[2 * i + 2]]
        random.shuffle(test_pos)
        for i in range(len(test_pos) - 1):
            test_list = insert_before_index(
                test_list,
                test_list.index(test_pos[i][1]),
                list(range(test_pos[i][0], test_pos[i][1])))
        self.assertEqual(
            origin,
            insert_before_index(
                test_list,
                test_list.index(
                    test_pos[len(test_pos) - 1][1]),
                list(range(test_pos[len(test_pos) - 1][0],
                           test_pos[len(test_pos) - 1][1]))))

    def test_insert_after_indices(self):
        self.assertEqual(
            [1, 0, 2, 3, 4],
            insert_after_indices([1, 2, 3, 4], [0], [0]))
        self.assertEqual(
            [1, 2, 3, 4, 0, 1, 2],
            insert_after_indices([1, 2, 3, 4], [3], [[0, 1, 2]]))
        self.assertEqual(
            [1, 2, 3, 4, 5],
            insert_after_indices([1, 2, 3, 4], [3], [5]))

        # Test successive replacement of individual elements
        num = random.randint(200, 1000)
        n = random.randint(20, 99)
        pos = random.sample(list(range(1, num - 1)), n)
        pos += [0, num]
        pos.sort()
        n += 2

        if n % 2 == 0:
            pos = pos[:-1]
            n -= 1
            pos[n - 1] = num
        test_pos = [[pos[i * 2 + 1], pos[i * 2 + 2]] for i in range(int(n / 2))]
        test_list = []
        origin = list(range(num))

        for i in range(int(n / 2)):
            test_list += origin[pos[2 * i]:pos[2 * i + 1]]
        random.shuffle(test_pos)
        self.assertEqual(
            origin,
            insert_after_indices(
                test_list,
                [test_list.index(item[0] - 1) for item in test_pos],
                [list(range(item[0], item[1])) for item in test_pos]))

    def test_insert_after_index(self):
        self.assertEqual(
            [1, 0, 2, 3, 4], insert_after_index([1, 2, 3, 4], 0, [0]))
        self.assertEqual(
            [1, 0, 2, 3, 4], insert_after_index([1, 2, 3, 4], 0, 0))
        self.assertEqual(
            [1, 2, 3, 4, 0, 1, 2],
            insert_after_index([1, 2, 3, 4], 3, [0, 1, 2]))
        self.assertEqual(
            [1, 2, 3, 4, 5], insert_after_index([1, 2, 3, 4], 3, [5]))

        # Test successive replacement of individual elements
        # TODO merge
        num = random.randint(200, 1000)
        n = random.randint(20, 99)
        pos = random.sample(list(range(1, num - 1)), n)
        pos += [0, num]
        n += 2
        pos.sort()

        if n % 2 == 0:
            pos = pos[:-1]
            n -= 1
            pos[n - 1] = num
        test_pos = [[pos[i * 2 + 1], pos[i * 2 + 2]] for i in range(int(n / 2))]
        test_list = []
        origin = list(range(num))

        for i in range(int(n / 2)):
            test_list += origin[pos[2 * i]:pos[2 * i + 1]]
        random.shuffle(test_pos)
        for i in range(len(test_pos) - 1):
            test_list = insert_after_index(
                test_list,
                test_list.index(test_pos[i][0] - 1),
                list(range(test_pos[i][0], test_pos[i][1])))
        self.assertEqual(
            origin,
            insert_after_index(
                test_list,
                test_list.index(test_pos[len(test_pos) - 1][0] - 1),
                list(range(test_pos[len(test_pos) - 1][0],
                           test_pos[len(test_pos) - 1][1]))))

    def test_swap_at_index(self):
        self.assertEqual([2, 1, 3, 4], swap_at_index([1, 2, 3, 4], 0, 1))
        self.assertEqual([4, 2, 3, 1], swap_at_index([1, 2, 3, 4], 3, 0))
        self.assertEqual([1, 3, 2, 4], swap_at_index([1, 2, 3, 4], 1, 2))
        num = random.randint(200, 1000)
        test_list = list(range(num))
        random.shuffle(test_list)

        for i in range(num):
            if test_list[i] == i:
                continue
            while test_list[i] != i:
                test_list = swap_at_index(test_list, test_list[i], i)
        self.assertEqual(list(range(num)), test_list)

    def test_test_descartes(self):
        self.assertEqual([[1]], descartes([[1]], 9))
        self.assertEqual([[1, 2]], descartes([[1], [2]], 9))

        num = 100
        test_list = [random.sample(list(range(int(1e6))),
                                   random.randint(1, 1000)) for i in range(num)]

        n = random.randint(10, 100)
        descartes_list = descartes(test_list, n)

        for item in descartes_list:
            for i in range(len(item)):
                self.assertTrue(item[i] in test_list[i])

        self.assertEqual(n, len(descartes_list))

        for i in range(len(descartes_list)):
            for j in range(i + 1, len(descartes_list)):
                self.assertTrue(descartes_list[i] != descartes_list[j])

    def test_unequal_replace_at_scopes(self):
        self.assertEqual(
            [1, 2], unequal_replace_at_scopes([1, 2, 3, 4], [[2, 5]], [[]]))
        self.assertEqual(
            [], unequal_replace_at_scopes([1, 2, 3, 4], [[0, 9]], [[]]))
        self.assertEqual(
            [4], unequal_replace_at_scopes([1, 2, 3, 4], [[0, 3]], [[]]))

        num = random.randint(200, 1000)

        test_list = random.sample(list(range(1, int(1e5))), num)
        test_list.sort()
        test_pos = random.sample(list(range(num)), num)
        change = []
        for i in range(num):
            if test_pos[i]:
                change.append(list(range(test_list[test_pos[i] - 1],
                                         test_list[test_pos[i]])))
            else:
                change.append(list(range(test_list[test_pos[i]])))
        change[test_pos.index(num - 1)] += list(range(test_list[num - 1],
                                                      int(1e5)))

        self.assertEqual(
            list(range(int(1e5))),
            unequal_replace_at_scopes(test_list, test_pos, change))


class TestListOpError(unittest.TestCase):
    def test_normalize_scope(self):
        self.assertRaises(TypeError, normalize_scope, None)
        self.assertRaises(TypeError, normalize_scope, TestClass)
        self.assertRaises(TypeError, normalize_scope, [0.9])

        for i in range(100):
            b = random.randint(0, 1e8)
            a = random.randint(1e9, 1e10)
            self.assertRaises(ValueError, normalize_scope, [a, b])
            self.assertRaises(ValueError, normalize_scope, (a, b))
            self.assertRaises(ValueError, normalize_scope, slice(a, b))
            self.assertRaises(ValueError, normalize_scope, (a, b, a))

    def test_replace_at_scopes(self):
        self.assertRaises(AssertionError, replace_at_scopes, None, [1], [1])
        self.assertRaises(AssertionError,
                          replace_at_scopes, [1], TestClass, [1])
        self.assertRaises(AssertionError, replace_at_scopes, [1], [1], None)
        self.assertRaises(AssertionError, replace_at_scopes, [1], [0], [1, 2])
        self.assertRaises(AssertionError, replace_at_scopes,
                          [1], [(1, 2), [3, 4]], [[1, 2], [4, 5]])
        self.assertRaises(ValueError, replace_at_scopes, [1], [1], [1])
        self.assertRaises(AssertionError, replace_at_scopes, [1, 2, 3], [1], [])
        self.assertRaises(AssertionError, replace_at_scopes, [1, 2, 3], [], [1])
        self.assertRaises(ValueError, replace_at_scopes,
                          [1, 2, 3], [[2, 1]], [1])
        self.assertRaises(ValueError, replace_at_scopes,
                          [1, 2, 3], [[2, 1], []], [1, []])
        self.assertRaises(ValueError, replace_at_scopes, [1, 2, 3], [-1], [1])
        self.assertRaises(TypeError, replace_at_scopes, [1, 2, 3], [1e1], [1])
        self.assertRaises(TypeError, replace_at_scopes,
                          [1, 2, 3], [['123']], [1])

        # Test collision
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2],
                          [[0, 1], [0, 1]], [[0], [1]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[4, 5], [0, 3], [2, 3]], [[1], [1, 2, 3], [3]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[0, 4], [4, 5], [2, 3]], [[1, 2, 3, 4], [1], [3]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[0, 4], [0, 1], [4, 5]], [[1, 2, 3, 4], [3], [1]])

    def test_replace_at_scope(self):
        self.assertRaises(AssertionError,
                          replace_at_scope, None, [1], [1])
        self.assertRaises(TypeError, replace_at_scope, [1], TestClass, [1])
        self.assertRaises(IndexError, replace_at_scope, [1], [1], None)
        self.assertRaises(IndexError, replace_at_scope, [1], [0], [1, 2])
        self.assertRaises(TypeError, replace_at_scope,
                          [1], [(1, 2), [3, 4]], [[1, 2], [4, 5]])
        self.assertRaises(IndexError, replace_at_scope, [1], [1], [1])
        self.assertRaises(IndexError, replace_at_scope, [1, 2, 3], [1], [])
        self.assertRaises(IndexError, replace_at_scope, [1, 2, 3], [], [1])
        self.assertRaises(TypeError, replace_at_scope, [1, 2, 3], [[2, 1]], [1])
        self.assertRaises(TypeError, replace_at_scope,
                          [1, 2, 3], [[1, 2], []], [1, []])
        self.assertRaises(TypeError, replace_at_scopes,
                          [1, 2, 3], [['123']], [1])
        self.assertRaises(ValueError, replace_at_scopes, [1, 2, 3], [-1], [1])
        self.assertRaises(TypeError, replace_at_scopes, [1, 2, 3], [9.3], [1])

    def test_delete_at_scopes(self):
        self.assertRaises(TypeError, delete_at_scopes, None, [1])
        self.assertRaises(AssertionError, delete_at_scopes, [1], TestClass)
        self.assertRaises(ValueError, delete_at_scopes, [1], [7])
        self.assertRaises(IndexError, delete_at_scopes, [1], [[1]])
        self.assertRaises(ValueError, delete_at_scopes, [1, 2, 3], [])
        self.assertRaises(ValueError, delete_at_scopes, [], [1])
        self.assertRaises(TypeError, delete_at_scopes, [1, 2, 3], [['123']])
        self.assertRaises(ValueError, delete_at_scopes, [1, 2, 1], [1, -1])
        self.assertRaises(TypeError, delete_at_scopes, [1, 2, 1], [1, 0.9])
        # Test collision
        self.assertRaises(ValueError, delete_at_scopes,
                          [0, 1, 2], [[0, 1], [0, 1]])
        self.assertRaises(ValueError, delete_at_scopes,
                          [0, 1, 2, 3, 4], [[4, 5], [0, 3], [2, 3]])
        self.assertRaises(ValueError, delete_at_scopes,
                          [0, 1, 2, 3, 4], [[0, 4], [4, 5], [2, 3]])
        self.assertRaises(ValueError, delete_at_scopes,
                          [0, 1, 2, 3, 4], [[0, 4], [0, 1], [4, 5]])

    def test_delete_at_scope(self):
        self.assertRaises(TypeError, delete_at_scope, None, [1])
        self.assertRaises(TypeError, delete_at_scope, [1], TestClass)
        self.assertRaises(IndexError, delete_at_scope, [1], [7])
        self.assertRaises(TypeError, delete_at_scope, [1], [[1]])
        self.assertRaises(IndexError, delete_at_scope, [1, 2, 3], [])
        self.assertRaises(IndexError, delete_at_scope, [], [1])
        self.assertRaises(TypeError, delete_at_scope, [1, 2, 3], [['123']])
        self.assertRaises(ValueError, delete_at_scope, [1], 2)
        self.assertRaises(ValueError, delete_at_scope, [1], -1)

    def test_insert_before_indices(self):
        self.assertRaises(TypeError, insert_before_indices, None, [1], [1])
        self.assertRaises(AssertionError,
                          insert_before_indices, [1], TestClass, [1])
        self.assertRaises(AssertionError, insert_before_indices, [1], [1], None)
        self.assertRaises(AssertionError,
                          insert_before_indices, [1], [0], [1, 2])
        self.assertRaises(AssertionError,
                          insert_before_indices, [1], [(1, 2)], [1, 2])
        self.assertRaises(IndexError, insert_before_indices, [1], [1], [1])
        self.assertRaises(AssertionError,
                          insert_before_indices, [1, 2, 3], [1], [])
        self.assertRaises(AssertionError,
                          insert_before_indices, [1, 2, 3], [], [1])
        self.assertRaises(TypeError,
                          insert_before_indices, [1, 2, 3], [1, []], [1, []])
        self.assertRaises(TypeError,
                          insert_before_indices, [1, 2, 3], [['123']], [1])
        self.assertRaises(TypeError, insert_before_indices,[1, 2], [[1]], [[0]])
        self.assertRaises(ValueError,
                          insert_before_indices, [1, 2], [1, -1], [[0], [0]])
        # Test collision
        self.assertRaises(ValueError,
                          insert_after_indices, [0, 1, 2], [0, 0], [0, 1])
        self.assertRaises(ValueError,
                          insert_before_indices, [0, 1, 2], [2, 2], [1, 1])

    def test_insert_before_index(self):
        self.assertRaises(TypeError, insert_before_index, None, 1, [0])
        self.assertRaises(TypeError, insert_before_index, [1], TestClass, [0])
        self.assertRaises(TypeError, insert_before_index, [1], [0], [1, 2])
        self.assertRaises(TypeError, insert_before_index, [1], (1, 2), [1, 2])
        self.assertRaises(IndexError, insert_before_index, [1], 1, [1])
        self.assertRaises(TypeError,
                          insert_before_index, [1, 2, 3], [1, 2], [1])
        self.assertRaises(TypeError, insert_before_index, [1, 2, 3], [], [1])
        self.assertRaises(TypeError,
                          insert_before_index, [1, 2, 3], [1, []], [1])
        self.assertRaises(TypeError, insert_before_index, [1, 2, 3], '123', [1])
        self.assertRaises(ValueError, insert_before_index, [1, 2, 3], -1, [1])

    def test_insert_after_indices(self):
        # TODO merge with others
        self.assertRaises(TypeError, insert_after_indices, None, [1], [1])
        self.assertRaises(AssertionError,
                          insert_after_indices, [1], TestClass, [1])
        self.assertRaises(AssertionError, insert_after_indices, [1], [1], None)
        self.assertRaises(AssertionError,
                          insert_after_indices, [1], [0], [1, 2])
        self.assertRaises(AssertionError,
                          insert_after_indices, [1], [(1, 2)], [1, 2])
        self.assertRaises(IndexError, insert_after_indices, [1], [1], [1])
        self.assertRaises(AssertionError,
                          insert_after_indices, [1, 2, 3], [1], [])
        self.assertRaises(AssertionError,
                          insert_after_indices, [1, 2, 3], [], [1])
        self.assertRaises(TypeError,
                          insert_after_indices, [1, 2, 3], [1, []], [1, []])
        self.assertRaises(TypeError,
                          insert_after_indices, [1, 2, 3], [['123']], [1])
        self.assertRaises(TypeError, insert_after_indices, [1, 2], [[1]], [[0]])
        self.assertRaises(ValueError, insert_after_indices, [1, 2], [-1], [[0]])
        # Test collision
        self.assertRaises(ValueError,
                          insert_after_indices, [0, 1, 2], [0, 0], [0, 1])
        self.assertRaises(ValueError,
                          insert_after_indices, [0, 1, 2], [2, 2], [1, 1])

    def test_insert_after_index(self):
        # TODO meger with others
        self.assertRaises(TypeError, insert_after_index, None, 1, [0])
        self.assertRaises(TypeError, insert_after_index, [1], TestClass, [0])
        self.assertRaises(TypeError, insert_after_index, [1], [0], [1, 2])
        self.assertRaises(TypeError, insert_after_index, [1], (1, 2), [1, 2])
        self.assertRaises(IndexError, insert_after_index, [1], 1, [1])
        self.assertRaises(TypeError, insert_after_index, [1, 2, 3], [1, 2], [1])
        self.assertRaises(TypeError, insert_after_index, [1, 2, 3], [], [1])
        self.assertRaises(TypeError, insert_after_index, [1, 2, 3], [1, []], [1])
        self.assertRaises(TypeError, insert_after_index, [1, 2, 3], '123', [1])
        self.assertRaises(ValueError, insert_after_index, [1, 2, 3], -1, [1])

    def test_swap_at_index(self):
        self.assertRaises(TypeError, swap_at_index, None, 1, 0)
        self.assertRaises(TypeError, swap_at_index, [1, 2], TestClass, 0)
        self.assertRaises(TypeError, swap_at_index, [1, 2], 1, [0])
        self.assertRaises(ValueError, swap_at_index, [1], 0, 0)
        self.assertRaises(ValueError, swap_at_index, [1], 0, 1)
        self.assertRaises(ValueError, swap_at_index, [1, 2, 3], 0, 3)
        self.assertRaises(ValueError, swap_at_index, [1, 2, 3], 0, -1)

    def test_descartes(self):
        self.assertRaises(AssertionError, descartes, None, 1)
        self.assertRaises(ValueError, descartes, [], 1)
        self.assertRaises(ValueError, descartes, [[1, 2], []], 1)
        self.assertRaises(AssertionError, descartes, [[1], '123'], 1)
        self.assertRaises(AssertionError, descartes, [[1], [2]], TestClass)
        self.assertRaises(AssertionError, descartes, [[1], [2]], -1)
        self.assertRaises(ValueError, descartes, [[]], 1)
        self.assertRaises(AssertionError, descartes, [[1]], 0.9)

    def test_unequal_replace_at_scopes(self):
        self.assertRaises(TypeError,
                          unequal_replace_at_scopes, None, [1], [1])
        self.assertRaises(AssertionError,
                          unequal_replace_at_scopes, [1], TestClass, [1])
        self.assertRaises(AssertionError,
                          unequal_replace_at_scopes, [1], [1], None)
        self.assertRaises(AssertionError,
                          unequal_replace_at_scopes, [1], [0], [1, 2])
        self.assertRaises(ValueError,
                          unequal_replace_at_scopes,
                          [1], [(1, 2), [3, 4]], [[1, 2], [4, 5]])
        self.assertRaises(ValueError,
                          unequal_replace_at_scopes, [1], [1], [1])
        self.assertRaises(AssertionError,
                          unequal_replace_at_scopes, [1, 2, 3], [1], [])
        self.assertRaises(AssertionError,
                          unequal_replace_at_scopes, [1, 2, 3], [], [1])
        self.assertRaises(ValueError,
                          unequal_replace_at_scopes, [1, 2, 3], [[2, 1]], [1])
        self.assertRaises(ValueError,
                          unequal_replace_at_scopes,
                          [1, 2, 3], [[2, 1], []], [1, []])
        self.assertRaises(ValueError,
                          unequal_replace_at_scopes, [1, 2, 3], [-1], [1])
        self.assertRaises(TypeError,
                          unequal_replace_at_scopes, [1, 2, 3], [1e1], [1])
        self.assertRaises(TypeError,
                          unequal_replace_at_scopes, [1, 2, 3], [['123']], [1])

        # Test collision
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2],
                          [[0, 1], [0, 1]], [[0], [1]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[4, 5], [0, 3], [2, 3]], [[1], [1, 2, 3], [3]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[0, 4], [4, 5], [2, 3]], [[1, 2, 3, 4], [1], [3]])
        self.assertRaises(ValueError, replace_at_scopes, [0, 1, 2, 3, 4],
                          [[0, 4], [0, 1], [4, 5]], [[1, 2, 3, 4], [3], [1]])


if __name__ == "__main__":
    unittest.main()
