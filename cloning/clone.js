import fs from "fs";
import http from "isomorphic-git/http/node";
import git from "isomorphic-git";
import path from "path";

const repoUrl = process.argv[2];        // full repo URL from Python
const localDir = "./models";

if (!repoUrl) {
  console.error("Missing repo URL argument");
  process.exit(1);
}

const repoName = repoUrl.split("/").slice(-1)[0].replace(".git", "");
const target = path.join(localDir, repoName);

(async () => {
  try {
    await fs.promises.mkdir(target, { recursive: true });
    await git.clone({
      fs,
      http,
      dir: target,
      url: repoUrl,
      singleBranch: true,
      depth: 1,
    });
    console.log(`Cloned ${repoUrl} into ${target}`);
  } catch (err) {
    console.error("Clone failed:", err.message);
    process.exit(1);
  }
})();