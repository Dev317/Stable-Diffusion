/** @type {import('next').NextConfig} */
const nextConfig = {
	experimental: {
		appDir: true,
	},
	images: {
		remotePatterns: [
			{ protocol: "http", hostname: '*' },
			{ protocol: "https", hostname: '*' },
		]
	}
};

module.exports = nextConfig;
