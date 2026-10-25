//! Visiting a range of a latched tree's entries without latching their
//! leaves, for keys and values that are copies of their bytes: `Copy`, and
//! owning nothing. As a lookup copies one value unlatched (`lookup.rs`,
//! D57), a copied scan copies a leaf's entries at the version it read the
//! leaf at, and checks the version after. A leaf that changed meanwhile
//! sends the rest of the scan the latched way (`scan.rs`).
//!
//! - **A leaf is copied in chunks** of a few entries, each checked by the
//!   leaf's version before the closure sees it: a scan that stops early
//!   copies little, and a chunk lives on the stack whatever the leaf's
//!   capacity.
//! - **Leaf to leaf, by version.** The next leaf is found down the internal
//!   nodes the descent kept, as a latched scan finds it, read at a version
//!   instead of latched. Once its version is read, the previous leaf is
//!   checked again. Unchanged since its copies were checked, no entry moved
//!   into or out of it meanwhile: the next leaf is still its neighbor, and
//!   holds nothing the scan visited. Splits, merges and moves between
//!   siblings hold both leaves exclusively and bump both versions
//!   (`held.rs`). A check that fails starts over, from past the last key
//!   visited.
//! - **Not a snapshot**, as a latched scan isn't: an entry inserted or
//!   removed while the scan runs may or may not be visited. An entry present
//!   for the whole scan is visited exactly once: the scan goes on only past
//!   keys it visited from copies whose checks held.
//!
//! No latch is taken on this path, so scanners write nothing to the leaves
//! they read: a leaf's latch word stays shared between the cores scanning
//! it, where a latched scan writes it twice a leaf
//! (`design/09-concurrency.md`, principle 3).

use core::borrow::Borrow;
use core::cmp::Ordering as KeyOrdering;
use core::mem::MaybeUninit;
use core::ops::{Bound, ControlFlow};

use super::super::node::inline_keys::SearchKey;
use super::super::node::latch::{Latched, Version};
use super::super::node::leaf::Leaf;
use super::super::node::optimistic::RacingEntry;
use super::LatchedTree;
use super::descent::{Optimistic, TreePath};
use super::scan::{Ascending, NextLeaf};
use crate::inline_key::InlineKeyStrategy;

/// How many entries a copy takes at once.
const CHUNK: usize = 32;

/// How visiting a leaf's copies ended.
enum Visited<K, R> {
    /// The closure broke with this.
    Broke(R),
    /// A key past the bound the scan stops at.
    PastEnd,
    /// Every entry of the leaf was visited; the last key visited, if any.
    Exhausted(Option<K>),
    /// The leaf changed while a chunk was read, and the chunk was thrown
    /// away; the last key visited before it, if any.
    Changed(Option<K>),
}

/// The leaf a copied scan starts in, read at a version.
enum FirstLeaf<K, V, S: InlineKeyStrategy<K>> {
    At(Leaf<K, V, Latched<S>>, Version),
    /// The range starts after every key.
    None,
}

impl<K: Copy, V: Copy, S: InlineKeyStrategy<K>> LatchedTree<K, V, S> {
    /// Runs `f` on a copy of every entry whose key lies between `start` and
    /// `end`, in key order, without latching their leaves where their
    /// versions hold (module documentation). Returns what `f` broke with,
    /// or `None` once past the range.
    pub(crate) fn scan_copied<Q, R>(
        &self,
        start: Bound<&Q>,
        end: Bound<&Q>,
        mut f: impl FnMut(&K, &V) -> ControlFlow<R>,
    ) -> Option<R>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.root_page()?;
        let mut path = TreePath::new();
        // The last key visited, which the scan goes on past when it starts
        // over, or falls back to latching.
        let mut last: Option<K> = None;
        'restart: loop {
            let resume = last;
            let from = match &resume {
                Some(key) => Bound::Excluded(key.borrow()),
                None => start,
            };
            let (mut leaf, mut version) = match self.first_leaf_racing::<Q>(from, &mut path) {
                FirstLeaf::At(leaf, version) => (leaf, version),
                FirstLeaf::None => return None,
            };
            let mut first = true;
            loop {
                let from = if first { from } else { Bound::Unbounded };
                match self.visit_copied::<Q, R>(leaf, version, from, end, &mut f) {
                    Visited::Broke(result) => return Some(result),
                    Visited::PastEnd => return None,
                    Visited::Changed(seen) => {
                        last = seen.or(last);
                        return self.scan_latched_past(last, start, end, f);
                    }
                    Visited::Exhausted(seen) => last = seen.or(last),
                }
                first = false;
                match self.next_leaf::<Ascending, Optimistic>(&mut path) {
                    NextLeaf::None => return None,
                    NextLeaf::Latched(next) => {
                        // Checked once the next leaf's version is read: the
                        // previous leaf unchanged since its copies were
                        // checked, nothing moved between the two meanwhile
                        // (module documentation).
                        if !leaf.latch().is_unchanged(version) {
                            continue 'restart;
                        }
                        (leaf, version) = (next.leaf(), next.version());
                    }
                    NextLeaf::StartOver => continue 'restart,
                }
            }
        }
    }

    /// The leaf where a copied scan from `from` starts, read at a version,
    /// with `path` keeping the internal nodes down to it.
    fn first_leaf_racing<Q>(
        &self,
        from: Bound<&Q>,
        path: &mut TreePath<K, V, S>,
    ) -> FirstLeaf<K, V, S>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let key = match from {
            Bound::Unbounded => return self.leftmost_leaf_racing(path),
            Bound::Included(key) | Bound::Excluded(key) => SearchKey::new(key),
        };
        match self.descend::<Q, Optimistic, _>(&key, path) {
            Ok((held, _)) => FirstLeaf::At(held.leaf(), held.version()),
            // The bound lacks what every key shares and sorts before them
            // all: the scan starts at the first leaf, and compares each of
            // its entries with the bound. A shrink of the shared prefix
            // since, which may have put keys before the bound, holds the
            // border leaves and bumps their versions: a copy of the first
            // leaf from before it then fails its check.
            Err(KeyOrdering::Less) => self.leftmost_leaf_racing(path),
            Err(_) => FirstLeaf::None,
        }
    }

    /// The tree's first leaf, read at a version.
    fn leftmost_leaf_racing(&self, path: &mut TreePath<K, V, S>) -> FirstLeaf<K, V, S> {
        let held = self.first_leaf::<Ascending, Optimistic>(path);
        FirstLeaf::At(held.leaf(), held.version())
    }

    /// Runs `f` on copies of the entries of `leaf`, read at `version`, from
    /// `from` to `to`, a chunk at a time, each chunk checked by the version
    /// before `f` sees it.
    fn visit_copied<Q, R>(
        &self,
        leaf: Leaf<K, V, Latched<S>>,
        version: Version,
        from: Bound<&Q>,
        to: Bound<&Q>,
        f: &mut impl FnMut(&K, &V) -> ControlFlow<R>,
    ) -> Visited<K, R>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let page = leaf.node_ptr();
        // SAFETY: the page is a leaf's of a latched tree, allocated while
        // threads share it, its header initialized.
        let (has_head, len) = unsafe { (page.load_racing_has_head(), page.load_racing_header().0) };
        let len = len.min(Leaf::<K, V, Latched<S>>::capacity());
        let mut chunk: [MaybeUninit<RacingEntry<K, V>>; CHUNK] =
            [const { MaybeUninit::uninit() }; CHUNK];
        let mut seen = None;
        // The head goes first in the first chunk, then the keys, in order.
        let mut head_pending = has_head;
        let mut next_key = 0;
        while head_pending || next_key < len {
            let mut filled = 0;
            if head_pending {
                match leaf.load_racing_head_entry() {
                    Some(head) => chunk[0].write(head),
                    // The head went meanwhile, and so did the version.
                    None => return Visited::Changed(seen),
                };
                filled = 1;
                head_pending = false;
            }
            let taking = (len - next_key).min(CHUNK - filled);
            leaf.load_racing_entries(next_key, &mut chunk[filled..filled + taking]);
            filled += taking;
            next_key += taking;
            if !leaf.latch().is_unchanged(version) {
                return Visited::Changed(seen);
            }
            for slot in &chunk[..filled] {
                // SAFETY: the slot was written above. The leaf stayed at
                // `version`, with no thread holding it exclusively, from
                // before the reads to after them (checked just above), and
                // each read was of its head, which it had, or of a key below
                // its length as read. The copies are of `Copy` keys and
                // values, which own nothing, so they stand once the leaf
                // changes.
                let (key, value) = unsafe { slot.assume_init_read().assume_read() };
                let (key, value) = (*key, *value);
                if !Self::past_start(&key, from) {
                    continue;
                }
                if Self::past_end(&key, to) {
                    return Visited::PastEnd;
                }
                seen = Some(key);
                if let ControlFlow::Break(result) = f(&key, &value) {
                    return Visited::Broke(result);
                }
            }
        }
        Visited::Exhausted(seen)
    }

    /// The rest of a scan, the latched way (`scan.rs`): from past `last`,
    /// or from `start` without one, to `end`.
    #[cold]
    #[inline(never)]
    fn scan_latched_past<Q, R>(
        &self,
        last: Option<K>,
        start: Bound<&Q>,
        end: Bound<&Q>,
        mut f: impl FnMut(&K, &V) -> ControlFlow<R>,
    ) -> Option<R>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let from = match &last {
            Some(key) => Bound::Excluded(key.borrow()),
            None => start,
        };
        self.scan(from, end, None, |key, value, _| f(key, value))
    }

    /// Whether `key` is at or past the bound a scan starts from.
    fn past_start<Q>(key: &K, from: Bound<&Q>) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match from {
            Bound::Included(from) => key.borrow() >= from,
            Bound::Excluded(from) => key.borrow() > from,
            Bound::Unbounded => true,
        }
    }

    /// Whether `key` is past the bound a scan stops at.
    fn past_end<Q>(key: &K, to: Bound<&Q>) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match to {
            Bound::Included(to) => key.borrow() > to,
            Bound::Excluded(to) => key.borrow() >= to,
            Bound::Unbounded => false,
        }
    }
}
