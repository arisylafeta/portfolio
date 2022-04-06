import Head from 'next/head';

const CustomHead = ({ title }) => {
  return (
    <Head>
      <title> {title} </title>
      <meta
        name="description"
        content="Arianit Sylafeta is an avid full stack web developer building websites and applications you'd love to use"
      />
      <meta
        name="keywords"
        content="ari sylafeta, ari, sylafeta, web developer portfolio, ari web developer, ari developer, mern stack, ari sylafeta portfolio, vscode-portfolio"
      />
      <meta property="og:title" content="Arianit Sylafeta's Portfolio" />
      <meta
        property="og:description"
        content="A full-stack developer building websites that you'd like to use."
      />
      <meta property="og:image" content="https://imgur.com/QjyRAiQ.png" />
      <meta property="og:url" content="https://www.arisylafeta.com" />
      <meta name="twitter:card" content="summary_large_image" />
    </Head>
  );
};

export default CustomHead;

CustomHead.defaultProps = {
  title: 'Arianit Sylafeta',
};
