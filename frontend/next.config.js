/** @type {import('next').NextConfig} */
const nextConfig = {
  // Compile to plain static files so FastAPI can serve them.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
};

module.exports = nextConfig;
