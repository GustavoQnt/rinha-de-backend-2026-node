// Parse TIMING lines and print percentiles per phase (fraud only).
// Usage: node analyze-timing.cjs file1 [file2 ...]
const fs = require('fs');
const files = process.argv.slice(2);
const pre = [], predict = [], post = [];
for (const f of files) {
    for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
        const m = line.match(/^TIMING fraud pre=(\d+) predict=(\d+) post=(\d+)/);
        if (!m) continue;
        pre.push(+m[1]); predict.push(+m[2]); post.push(+m[3]);
    }
}
const sortn = a => a.slice().sort((x, y) => x - y);
const p = (a, q) => a[Math.floor((a.length - 1) * q)];
const fmt = n => `${(n/1000).toFixed(1)}µs`;
const stats = (name, arr) => {
    const s = sortn(arr);
    console.log(
        `${name.padEnd(8)} n=${s.length}  min=${fmt(s[0])}  p50=${fmt(p(s,.5))}  p90=${fmt(p(s,.9))}  p99=${fmt(p(s,.99))}  p999=${fmt(p(s,.999))}  max=${fmt(s.at(-1))}`
    );
};
stats('PRE', pre);
stats('PREDICT', predict);
stats('POST', post);
const sumP99 = p(sortn(pre), .99) + p(sortn(predict), .99) + p(sortn(post), .99);
console.log(`\nNaive sum of p99s ≈ ${fmt(sumP99)} (predict share ≈ ${(p(sortn(predict),.99)/sumP99*100).toFixed(1)}%)`);
