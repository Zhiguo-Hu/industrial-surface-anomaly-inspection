"""Train-only positive-tile policies; negative and augmentation RNG stays separate."""
import hashlib
import random
import numpy as np

POLICIES=('uniform','class-balanced','class-small')
SMALL_MAX_PIXELS=int(256*256*.01)


class ClassAwareSampler:
    def __init__(self,tile_classes,foreground_pixels,policy,seed,minimum_small_tiles=20,minimum_small_parents=10,tiles_per_parent=12):
        if policy not in POLICIES[1:]:raise ValueError('Expected class-balanced or class-small')
        if min(minimum_small_tiles,minimum_small_parents,tiles_per_parent)<1:raise ValueError('Invalid small-pool guard')
        if set(tile_classes)!=set(foreground_pixels):raise ValueError('Tile metadata identities differ')
        if any(n<0 or n>256*256 or bool(n)!=bool(tile_classes[i]) for i,n in foreground_pixels.items()):
            raise ValueError('Inconsistent tile foreground and class labels')
        self.policy=policy;self.rng=random.Random(seed);self.area_rng=random.Random(seed+10000)
        self.pools={c:sorted(i for i,cs in tile_classes.items() if c in cs) for c in (1,2,3,4)}
        if any(not p for p in self.pools.values()):raise ValueError('All four training classes need positive tiles')
        self.small={c:[i for i in p if foreground_pixels[i]<=SMALL_MAX_PIXELS] for c,p in self.pools.items()}
        self.small_parents={c:len({i//tiles_per_parent for i in p}) for c,p in self.small.items()}
        self.eligible={c:len(self.small[c])>=minimum_small_tiles and self.small_parents[c]>=minimum_small_parents for c in self.pools}
        self.guard={'minimum_small_tiles':minimum_small_tiles,'minimum_small_parents':minimum_small_parents,
                    'tiles_per_parent':tiles_per_parent}

    def draw(self):
        result=[]
        for c in (1,2,3,4):
            prefer_small=self.area_rng.random()<.5
            pool=self.small[c] if self.policy=='class-small' and prefer_small and self.eligible[c] else self.pools[c]
            result.append(self.rng.choice(pool))
        return result

    def inventory(self):
        return {'policy':self.policy,'positive_target_classes_per_batch':[1,2,3,4],
                'small_definition':'0 < union-mask pixels <= floor(256*256*0.01); tile area, not parent-image area',
                'small_max_pixels':SMALL_MAX_PIXELS,'prefer_small_probability':.5 if self.policy=='class-small' else 0,
                'small_pool_guard':self.guard,'insufficient_small_pool_fallback':'uniform within the same class',
                'per_class':{str(c):{'positive_tiles':len(self.pools[c]),'small_tiles':len(self.small[c]),
                                     'small_parents':self.small_parents[c],'small_bias_eligible':self.eligible[c]} for c in self.pools}}


def update_negative_flip_hash(digest,negative,horizontal,vertical):
    digest.update(np.asarray(negative,dtype='<i8').tobytes());digest.update(bytes([horizontal,vertical]))


def reference_negative_flip_hash(positive,negative,seed,steps):
    rng=random.Random(seed);digest=hashlib.sha256()
    for _ in range(steps):
        rng.choices(positive,k=4)
        negatives=rng.choices(negative,k=4)
        update_negative_flip_hash(digest,negatives,rng.random()<.5,rng.random()<.5)
    return digest.hexdigest()
