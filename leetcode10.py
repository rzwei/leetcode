class TreeNode:
    def __init__(self, s, e):
        self.start = s
        self.end = e
        self.right = None
        self.left = None


class MyCalendar:
    def __init__(self):
        self.times = {}
        self.books = TreeNode(-1, -1)

    def search(self, p, s, e):
        if p.start >= e:
            if not p.left:
                p.left = TreeNode(s, e)
                return True
            else:
                return self.search(p.left, s, e)
        if p.end <= s:
            if not p.right:
                p.right = TreeNode(s, e)
                return True
            else:
                return self.search(p.right, s, e)
        return False

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        return self.search(self.books, start, end)


# class MyCalendar:
#     def __init__(self):
#         self.start = []
#         self.end = []
#
#     def book(self, start, end):
#         i = bisect.bisect_right(self.end, start)
#         j = bisect.bisect_left(self.start, end)
#         if i == j:
#             self.start.index(i, start)
#             self.end.index(i, end)
#             return True
#         return False


class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        # 734. Sentence Similarity
        if len(words1) != len(words2):
            return False
        d = {}
        for l, r in pairs:
            if l not in d:
                d[l] = {r}
            else:
                d[l].add(r)
            if r not in d:
                d[r] = {l}
            else:
                d[r].add(l)
        for w1, w2 in zip(words1, words2):
            if w1 == w2 or w1 in d.get(w2, []):
                continue
            return False
        return True

    def areSentencesSimilarTwo(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        if len(words1) != len(words2):
            return False
        d = {}

        def father(x):
            if x != d[x]:
                return father(d[x])
            else:
                return x

        def union(x, y):
            x = father(x)
            y = father(y)
            d[x] = y

        for w1, w2 in pairs:
            if w1 not in d:
                d[w1] = w1
            if w2 not in d:
                d[w2] = w2
            union(w1, w2)
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                continue
            if w1 not in d or w2 not in d:
                return False
            if father(w1) == father(w2):
                continue
            print(w1, w2)
            return False
        return True


if __name__ == '__main__':
    sol = Solution()

    print(sol.areSentencesSimilarTwo(["great", "acting", "skills"],
                                     ["fine", "painting", "talent"],
                                     [["great", "fine"], ["drama", "acting"], ["skills", "talent"]]))

    print(sol.areSentencesSimilarTwo(['great', 'acting', 'skills'], ['fine', 'drama', 'talent'],
                                     [["great", "fine"], ["acting", "drama"], ["skills", "talent"]]))
    # calendar = MyCalendar()
    # for v1, v2 in [[97, 100], [33, 51], [89, 100], [83, 100], [75, 92], [76, 95], [19, 30], [53, 63], [8, 23], [18, 37],
    #                [87, 100], [83, 100], [54, 67], [35, 48], [58, 75], [70, 89], [13, 32], [44, 63], [51, 62], [2, 15]]:
    #     print(calendar.book(v1, v2))
    # print(calendar.times.items())
    words1 = ["jrocadcojmybpxmuj", "livgsrfvgtovcurzq", "mnrdscqkycodx", "wgcjlntupylayse", "tglnshmqlmkqqfbpf",
              "uzlxmaoro", "narvuaqmmkqhd", "xozoyaqxtbustrymo", "jrocadcojmybpxmuj", "ainlwrwabqcwq",
              "qnjidlmwmxxjgntez", "bbchthovla", "vaufbmwdrupcxpg", "zwwgloilddclufwze", "tyxrlpmcy", "wtjtdrlm",
              "edurtetzseifez", "yzxogkunvohdmro", "livgsrfvgtovcurzq", "wmpvjvzljhnaxvp", "rqbswlkw",
              "umlzibkkpsyvpdol", "jkcmceinlyhi", "wlvmfxbleuot", "aeloeauxmc", "ooyllkxg", "wlvmfxbleuot", "cuewcvuy",
              "vaufbmwdrupcxpg", "bbchthovla", "arigdtezmyz", "yzxogkunvohdmro", "wrszraxxdum", "dhmiuqhqlsprxy",
              "xpmxtfyvjrnujyxjh", "bfxbncez", "cjjkmybleu", "mnrdscqkycodx", "mzfpofjn", "livgsrfvgtovcurzq",
              "shfzcyboj", "xozoyaqxtbustrymo", "xozoyaqxtbustrymo", "orlzzpytpzazxr", "filnwifbukdqijgr",
              "fllqjtnxwmfoou", "mkmawbogphdttd", "rthpxoxyyiy", "dkhfozltuckwog", "wmpvjvzljhnaxvp", "dhmiuqhqlsprxy",
              "yltljjairlkrmdq", "cuewcvuy", "subzoyxjkfiwmfb", "mzvbgcizeeth", "narvuaqmmkqhd", "tglnshmqlmkqqfbpf",
              "rpesfkhfjucj", "xrgfejybbkezgor", "vaufbmwdrupcxpg", "czlgbqzffodsoxng", "suvvqdiceuogcmv",
              "fllqjtnxwmfoou", "yltljjairlkrmdq", "bubwouozgs", "mnrdscqkycodx", "rqbswlkw", "ooyllkxg",
              "livgsrfvgtovcurzq", "rthpxoxyyiy", "pyzcbpjhntpefbq", "wtjtdrlm", "rztcppnmud", "inuzvkgolupxelcal",
              "pdxsxjop", "wmpvjvzljhnaxvp", "xydwvemqvtgvzl", "hqpnoczciajvkbdy", "rvihrzzkt", "jzquemjzpvfbka",
              "gkqrglav", "qyaxqaqxiwr", "mzvbgcizeeth", "umlzibkkpsyvpdol", "vaufbmwdrupcxpg", "ooyllkxg",
              "arigdtezmyz", "bubwouozgs", "wtjtdrlm", "xozoyaqxtbustrymo", "jrocadcojmybpxmuj", "rnlryins",
              "fllqjtnxwmfoou", "livgsrfvgtovcurzq", "czlgbqzffodsoxng", "hlcsiukaroscfg", "bfxbncez", "ainlwrwabqcwq",
              "vaufbmwdrupcxpg", "vaufbmwdrupcxpg"]
    words2 = ["jrocadcojmybpxmuj", "livgsrfvgtovcurzq", "mnrdscqkycodx", "wgcjlntupylayse", "bbchthovla", "bfxbncez",
              "ztisufueqzequ", "yutahdply", "suvvqdiceuogcmv", "ainlwrwabqcwq", "fquzrlhdsnuwhhu", "tglnshmqlmkqqfbpf",
              "vaufbmwdrupcxpg", "zwwgloilddclufwze", "livgsrfvgtovcurzq", "wtjtdrlm", "edurtetzseifez",
              "ecqfdkebnamkfglk", "livgsrfvgtovcurzq", "wmpvjvzljhnaxvp", "ryubcgbzmxc", "pzlmeboecybxmetz",
              "hqpnoczciajvkbdy", "xpmxtfyvjrnujyxjh", "zwwgloilddclufwze", "khcyhttaaxp", "wlvmfxbleuot",
              "jzquemjzpvfbka", "vaufbmwdrupcxpg", "tglnshmqlmkqqfbpf", "mzvbgcizeeth", "cjjkmybleu", "orlzzpytpzazxr",
              "dhmiuqhqlsprxy", "mzfpofjn", "bfxbncez", "inuzvkgolupxelcal", "inhzsspqltvl", "wlvmfxbleuot",
              "livgsrfvgtovcurzq", "orlzzpytpzazxr", "yutahdply", "yutahdply", "orlzzpytpzazxr", "gdziaihbagl",
              "yltljjairlkrmdq", "mkmawbogphdttd", "aotjpvanljxe", "aeloeauxmc", "wmpvjvzljhnaxvp", "dhmiuqhqlsprxy",
              "yltljjairlkrmdq", "dnaaehrekqms", "khcyhttaaxp", "mzvbgcizeeth", "narvuaqmmkqhd", "rvihrzzkt",
              "bfufqsusp", "xrgfejybbkezgor", "vaufbmwdrupcxpg", "czlgbqzffodsoxng", "jrocadcojmybpxmuj",
              "yltljjairlkrmdq", "yltljjairlkrmdq", "bubwouozgs", "inhzsspqltvl", "bsybvehdny", "subzoyxjkfiwmfb",
              "livgsrfvgtovcurzq", "stkglpqdjzxmnlito", "evepphnzuw", "xrgfejybbkezgor", "rztcppnmud", "cjjkmybleu",
              "qyaxqaqxiwr", "ibwfxvxswjbecab", "xydwvemqvtgvzl", "hqpnoczciajvkbdy", "tglnshmqlmkqqfbpf",
              "dnaaehrekqms", "gkqrglav", "bfxbncez", "qvwvgzxqihvk", "umlzibkkpsyvpdol", "vaufbmwdrupcxpg",
              "khcyhttaaxp", "arigdtezmyz", "bubwouozgs", "fllqjtnxwmfoou", "xozoyaqxtbustrymo", "jrocadcojmybpxmuj",
              "rnlryins", "wtjtdrlm", "livgsrfvgtovcurzq", "gkqrglav", "orileazg", "uzlxmaoro", "ainlwrwabqcwq",
              "vaufbmwdrupcxpg", "vaufbmwdrupcxpg"]
    pairs = [["yutahdply", "yutahdply"], ["xozoyaqxtbustrymo", "xozoyaqxtbustrymo"],
             ["xozoyaqxtbustrymo", "xozoyaqxtbustrymo"],
             ["yutahdply", "yutahdply"], ["bsybvehdny", "bsybvehdny"], ["pzlmeboecybxmetz", "pzlmeboecybxmetz"],
             ["rqbswlkw", "rqbswlkw"], ["ryubcgbzmxc", "ryubcgbzmxc"], ["umlzibkkpsyvpdol", "umlzibkkpsyvpdol"],
             ["bsybvehdny", "bsybvehdny"], ["rqbswlkw", "bsybvehdny"], ["pzlmeboecybxmetz", "bsybvehdny"],
             ["ryubcgbzmxc", "ryubcgbzmxc"], ["umlzibkkpsyvpdol", "ryubcgbzmxc"],
             ["hqpnoczciajvkbdy", "hqpnoczciajvkbdy"],
             ["vdjccijgqk", "vdjccijgqk"], ["rztcppnmud", "rztcppnmud"], ["jkcmceinlyhi", "hqpnoczciajvkbdy"],
             ["vdjccijgqk", "vdjccijgqk"], ["rztcppnmud", "vdjccijgqk"], ["hqpnoczciajvkbdy", "hqpnoczciajvkbdy"],
             ["jkcmceinlyhi", "hqpnoczciajvkbdy"], ["suvvqdiceuogcmv", "llrzqdnoxbscnkqy"],
             ["jrocadcojmybpxmuj", "jrocadcojmybpxmuj"], ["suvvqdiceuogcmv", "suvvqdiceuogcmv"],
             ["llrzqdnoxbscnkqy", "suvvqdiceuogcmv"], ["jrocadcojmybpxmuj", "jrocadcojmybpxmuj"],
             ["dnaaehrekqms", "dnaaehrekqms"],
             ["jzquemjzpvfbka", "muaskefecskjghzn"], ["muaskefecskjghzn", "iziepzqne"], ["cuewcvuy", "dnaaehrekqms"],
             ["iziepzqne", "iziepzqne"], ["muaskefecskjghzn", "iziepzqne"], ["jzquemjzpvfbka", "iziepzqne"],
             ["dnaaehrekqms", "dnaaehrekqms"], ["cuewcvuy", "dnaaehrekqms"], ["rpesfkhfjucj", "xpmxtfyvjrnujyxjh"],
             ["wlvmfxbleuot", "bfufqsusp"], ["xpmxtfyvjrnujyxjh", "mzfpofjn"], ["rpesfkhfjucj", "rpesfkhfjucj"],
             ["mzfpofjn", "rpesfkhfjucj"], ["xpmxtfyvjrnujyxjh", "rpesfkhfjucj"], ["bfufqsusp", "bfufqsusp"],
             ["wlvmfxbleuot", "bfufqsusp"], ["lkopigreodypvude", "lkopigreodypvude"], ["hjogoueazw", "hjogoueazw"],
             ["qvwvgzxqihvk", "qvwvgzxqihvk"], ["mzvbgcizeeth", "mzvbgcizeeth"], ["mzvbgcizeeth", "arigdtezmyz"],
             ["arigdtezmyz", "arigdtezmyz"], ["qvwvgzxqihvk", "arigdtezmyz"], ["mzvbgcizeeth", "arigdtezmyz"],
             ["lkopigreodypvude", "lkopigreodypvude"], ["hjogoueazw", "lkopigreodypvude"],
             ["tglnshmqlmkqqfbpf", "tglnshmqlmkqqfbpf"], ["bbchthovla", "bbchthovla"],
             ["rvihrzzkt", "tglnshmqlmkqqfbpf"],
             ["tglnshmqlmkqqfbpf", "tglnshmqlmkqqfbpf"], ["rvihrzzkt", "tglnshmqlmkqqfbpf"],
             ["bbchthovla", "bbchthovla"],
             ["filnwifbukdqijgr", "pkirimjwvyxs"], ["gdziaihbagl", "gdziaihbagl"], ["hlcsiukaroscfg", "hlcsiukaroscfg"],
             ["gdziaihbagl", "orileazg"], ["orileazg", "orileazg"], ["gdziaihbagl", "orileazg"],
             ["hlcsiukaroscfg", "orileazg"],
             ["pkirimjwvyxs", "pkirimjwvyxs"], ["filnwifbukdqijgr", "pkirimjwvyxs"], ["xrgfejybbkezgor", "wtjtdrlm"],
             ["yltljjairlkrmdq", "fllqjtnxwmfoou"], ["wtjtdrlm", "wtjtdrlm"], ["xrgfejybbkezgor", "wtjtdrlm"],
             ["fllqjtnxwmfoou", "fllqjtnxwmfoou"], ["yltljjairlkrmdq", "fllqjtnxwmfoou"],
             ["ecqfdkebnamkfglk", "gwkkpxuvgp"],
             ["inuzvkgolupxelcal", "inuzvkgolupxelcal"], ["hgxrhkanzvzmsjpzl", "gwkkpxuvgp"],
             ["cjjkmybleu", "cjjkmybleu"],
             ["yzxogkunvohdmro", "yzxogkunvohdmro"], ["inuzvkgolupxelcal", "yzxogkunvohdmro"],
             ["yzxogkunvohdmro", "yzxogkunvohdmro"], ["cjjkmybleu", "yzxogkunvohdmro"],
             ["ecqfdkebnamkfglk", "ecqfdkebnamkfglk"],
             ["gwkkpxuvgp", "ecqfdkebnamkfglk"], ["hgxrhkanzvzmsjpzl", "ecqfdkebnamkfglk"],
             ["qnjidlmwmxxjgntez", "hhrllhedyy"],
             ["vyrvelteblnqaabc", "qnjidlmwmxxjgntez"], ["wzflhbbgtc", "wzflhbbgtc"], ["rnlryins", "rnlryins"],
             ["fquzrlhdsnuwhhu", "zzdvolqtndzfjvqqr"], ["zzdvolqtndzfjvqqr", "bvxiilsnsarhsyl"],
             ["qnjidlmwmxxjgntez", "vyrvelteblnqaabc"], ["vyrvelteblnqaabc", "vyrvelteblnqaabc"],
             ["hhrllhedyy", "vyrvelteblnqaabc"], ["rnlryins", "vyrvelteblnqaabc"],
             ["fquzrlhdsnuwhhu", "zzdvolqtndzfjvqqr"],
             ["zzdvolqtndzfjvqqr", "zzdvolqtndzfjvqqr"], ["bvxiilsnsarhsyl", "zzdvolqtndzfjvqqr"],
             ["wzflhbbgtc", "zzdvolqtndzfjvqqr"], ["wgcjlntupylayse", "wgcjlntupylayse"],
             ["hgtyntdmrgjh", "hgtyntdmrgjh"],
             ["cemnayjhlnj", "cemnayjhlnj"], ["wgcjlntupylayse", "wgcjlntupylayse"],
             ["hgtyntdmrgjh", "wgcjlntupylayse"],
             ["cemnayjhlnj", "cemnayjhlnj"], ["aeloeauxmc", "aeloeauxmc"], ["zwwgloilddclufwze", "aeloeauxmc"],
             ["dkhfozltuckwog", "dwojnswr"], ["dkhfozltuckwog", "dkhfozltuckwog"], ["dwojnswr", "dkhfozltuckwog"],
             ["aeloeauxmc", "aeloeauxmc"], ["zwwgloilddclufwze", "aeloeauxmc"], ["aotjpvanljxe", "aotjpvanljxe"],
             ["stkglpqdjzxmnlito", "aotjpvanljxe"], ["zfmpxgrevxp", "aotjpvanljxe"], ["evepphnzuw", "evepphnzuw"],
             ["rthpxoxyyiy", "pyzcbpjhntpefbq"], ["aotjpvanljxe", "stkglpqdjzxmnlito"],
             ["stkglpqdjzxmnlito", "stkglpqdjzxmnlito"],
             ["zfmpxgrevxp", "stkglpqdjzxmnlito"], ["rthpxoxyyiy", "rthpxoxyyiy"], ["evepphnzuw", "rthpxoxyyiy"],
             ["pyzcbpjhntpefbq", "rthpxoxyyiy"], ["czlgbqzffodsoxng", "czlgbqzffodsoxng"], ["gkqrglav", "gkqrglav"],
             ["gkqrglav", "gkqrglav"], ["czlgbqzffodsoxng", "czlgbqzffodsoxng"], ["tyxrlpmcy", "tyxrlpmcy"],
             ["livgsrfvgtovcurzq", "livgsrfvgtovcurzq"], ["tyxrlpmcy", "tyxrlpmcy"],
             ["livgsrfvgtovcurzq", "livgsrfvgtovcurzq"],
             ["cufxsgbpjgqvk", "cufxsgbpjgqvk"], ["cufxsgbpjgqvk", "inhzsspqltvl"], ["shsgrqol", "shsgrqol"],
             ["mnrdscqkycodx", "mnrdscqkycodx"], ["inhzsspqltvl", "inhzsspqltvl"], ["cufxsgbpjgqvk", "inhzsspqltvl"],
             ["shsgrqol", "shsgrqol"], ["mnrdscqkycodx", "shsgrqol"], ["rphnhtvnihyfkrgv", "rphnhtvnihyfkrgv"],
             ["edurtetzseifez", "edurtetzseifez"], ["yykdqtkkdacpbwtbq", "yykdqtkkdacpbwtbq"],
             ["rphnhtvnihyfkrgv", "rphnhtvnihyfkrgv"], ["edurtetzseifez", "rphnhtvnihyfkrgv"],
             ["yykdqtkkdacpbwtbq", "yykdqtkkdacpbwtbq"], ["dhmiuqhqlsprxy", "dhmiuqhqlsprxy"],
             ["ztisufueqzequ", "ztisufueqzequ"],
             ["narvuaqmmkqhd", "narvuaqmmkqhd"], ["narvuaqmmkqhd", "narvuaqmmkqhd"], ["ztisufueqzequ", "narvuaqmmkqhd"],
             ["dhmiuqhqlsprxy", "dhmiuqhqlsprxy"], ["wmpvjvzljhnaxvp", "wmpvjvzljhnaxvp"],
             ["ibwfxvxswjbecab", "ibwfxvxswjbecab"],
             ["xydwvemqvtgvzl", "wmpvjvzljhnaxvp"], ["wmpvjvzljhnaxvp", "wmpvjvzljhnaxvp"],
             ["xydwvemqvtgvzl", "wmpvjvzljhnaxvp"],
             ["ibwfxvxswjbecab", "ibwfxvxswjbecab"], ["mkmawbogphdttd", "mkmawbogphdttd"],
             ["bubwouozgs", "mkmawbogphdttd"],
             ["ainlwrwabqcwq", "ainlwrwabqcwq"], ["mkmawbogphdttd", "mkmawbogphdttd"], ["bubwouozgs", "mkmawbogphdttd"],
             ["ainlwrwabqcwq", "ainlwrwabqcwq"], ["uzlxmaoro", "bfxbncez"], ["qyaxqaqxiwr", "qyaxqaqxiwr"],
             ["pdxsxjop", "pdxsxjop"], ["pdxsxjop", "pdxsxjop"], ["qyaxqaqxiwr", "pdxsxjop"], ["bfxbncez", "bfxbncez"],
             ["uzlxmaoro", "bfxbncez"], ["subzoyxjkfiwmfb", "subzoyxjkfiwmfb"], ["ooyllkxg", "ooyllkxg"],
             ["subzoyxjkfiwmfb", "khcyhttaaxp"], ["subzoyxjkfiwmfb", "subzoyxjkfiwmfb"],
             ["khcyhttaaxp", "subzoyxjkfiwmfb"],
             ["ooyllkxg", "ooyllkxg"], ["orlzzpytpzazxr", "orlzzpytpzazxr"], ["oufzmjgplt", "oufzmjgplt"],
             ["wrszraxxdum", "wrszraxxdum"], ["shfzcyboj", "shfzcyboj"], ["oufzmjgplt", "oufzmjgplt"],
             ["orlzzpytpzazxr", "oufzmjgplt"], ["wrszraxxdum", "wrszraxxdum"], ["shfzcyboj", "wrszraxxdum"],
             ["yutahdply", "xozoyaqxtbustrymo"], ["umlzibkkpsyvpdol", "pzlmeboecybxmetz"],
             ["hqpnoczciajvkbdy", "rztcppnmud"],
             ["llrzqdnoxbscnkqy", "jrocadcojmybpxmuj"], ["cuewcvuy", "jzquemjzpvfbka"],
             ["rpesfkhfjucj", "wlvmfxbleuot"],
             ["lkopigreodypvude", "mzvbgcizeeth"], ["tglnshmqlmkqqfbpf", "bbchthovla"],
             ["orileazg", "filnwifbukdqijgr"],
             ["yltljjairlkrmdq", "xrgfejybbkezgor"], ["inuzvkgolupxelcal", "hgxrhkanzvzmsjpzl"],
             ["hhrllhedyy", "wzflhbbgtc"],
             ["cemnayjhlnj", "hgtyntdmrgjh"], ["dkhfozltuckwog", "zwwgloilddclufwze"],
             ["zfmpxgrevxp", "pyzcbpjhntpefbq"],
             ["gkqrglav", "czlgbqzffodsoxng"], ["tyxrlpmcy", "livgsrfvgtovcurzq"], ["shsgrqol", "cufxsgbpjgqvk"],
             ["rphnhtvnihyfkrgv", "yykdqtkkdacpbwtbq"], ["dhmiuqhqlsprxy", "ztisufueqzequ"],
             ["ibwfxvxswjbecab", "xydwvemqvtgvzl"],
             ["mkmawbogphdttd", "ainlwrwabqcwq"], ["pdxsxjop", "uzlxmaoro"], ["ooyllkxg", "khcyhttaaxp"],
             ["shfzcyboj", "orlzzpytpzazxr"]]
    print(sol.areSentencesSimilarTwo(words1, words2, pairs))
